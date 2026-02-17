# Jules Task: Vectorized CPU Fallback for QuantumFieldEngine

## Problem
`core/quantum_field_engine.py` hard-crashes if CUDA is unavailable (`raise RuntimeError`). This makes the engine untestable on CPU-only machines and prevents graceful degradation. Additionally, the result-assembly loop in `batch_compute_states()` (lines 307-384) iterates one-by-one in Python to build `ThreeBodyQuantumState` objects — this is the remaining bottleneck after GPU compute.

## Solution
1. Add a `_batch_compute_cpu()` method that replicates the full CUDA physics pipeline using vectorized NumPy with `numpy.lib.stride_tricks.sliding_window_view` for rolling regression
2. Make `__init__` gracefully handle missing CUDA (`self.use_gpu = False` instead of crashing)
3. Update `batch_compute_states()` to dispatch to CPU path when GPU unavailable

## File: `core/quantum_field_engine.py`

### Step 1: Make `__init__` graceful

Replace lines 98-104:

```python
# Old:
if not cuda.is_available():
    raise RuntimeError("CUDA accelerator is mandatory but not available on this system.")

if not CUDA_PHYSICS_AVAILABLE:
    raise RuntimeError("core.cuda_physics module is missing.")

self.use_gpu = True

# New:
if cuda.is_available() and CUDA_PHYSICS_AVAILABLE:
    self.use_gpu = True
else:
    self.use_gpu = False
    print("WARNING: CUDA not available — using vectorized CPU fallback")
```

### Step 2: Add `_batch_compute_cpu()` method

Add this method to `QuantumFieldEngine`, after `batch_compute_states()`. It replicates the exact same physics as `compute_physics_kernel` and `detect_archetype_kernel` from `core/cuda_physics.py`, but uses vectorized NumPy.

```python
def _batch_compute_cpu(self, day_data: pd.DataFrame, params: dict = None) -> list:
    """
    Vectorized CPU fallback for batch_compute_states.
    Uses numpy.lib.stride_tricks.sliding_window_view for rolling regression.
    Replicates the exact physics from cuda_physics.py kernels.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    params = params or {}
    n = len(day_data)
    rp = self.regression_period

    if n < rp:
        return []

    # Extract input arrays
    prices = day_data['price'].values.astype(np.float64) if 'price' in day_data.columns else day_data['close'].values.astype(np.float64)
    volumes = day_data['volume'].values.astype(np.float64) if 'volume' in day_data.columns else np.zeros(n, dtype=np.float64)

    # ── 1. Rolling Linear Regression (vectorized) ──
    # Create rolling windows: shape (n - rp + 1, rp)
    windows = sliding_window_view(prices, rp)  # windows[i] = prices[i:i+rp]
    # Result index mapping: windows[i] corresponds to bar index i + rp - 1

    n_windows = windows.shape[0]
    x = np.arange(rp, dtype=np.float64)  # [0, 1, ..., rp-1]
    mean_x = x.mean()
    sum_x = x.sum()
    sum_xx = (x * x).sum()
    denom = sum_xx - sum_x * sum_x / rp

    # Vectorized sums over each window
    sum_y = windows.sum(axis=1)           # shape: (n_windows,)
    sum_xy = (windows * x).sum(axis=1)    # shape: (n_windows,)
    sum_yy = (windows * windows).sum(axis=1)

    mean_y = sum_y / rp

    # Slope, center, sigma
    inv_denom = 1.0 / denom if abs(denom) > 1e-9 else 0.0
    slope = (sum_xy - mean_x * sum_y) * inv_denom
    center = mean_y + slope * ((rp - 1) - mean_x)

    # Sigma (residual std)
    sst = sum_yy - rp * mean_y * mean_y
    rss = np.maximum(sst - slope * slope * denom, 0.0)
    sigma = np.where(rp > 2, np.sqrt(rss / (rp - 2)), 0.0)
    sigma = np.maximum(sigma, 1e-6)

    # Full-length arrays (first rp-1 bars get defaults)
    full_center = np.zeros(n)
    full_sigma = np.full(n, 1e-6)
    full_slope = np.zeros(n)
    full_center[rp-1:] = center
    full_sigma[rp-1:] = sigma
    full_slope[rp-1:] = slope

    # ── 2. Z-Score ──
    z_scores = np.zeros(n)
    z_scores[rp-1:] = (prices[rp-1:] - center) / sigma

    # ── 3. Velocity & Momentum ──
    velocity = np.zeros(n)
    velocity[1:] = np.diff(prices)
    momentum = np.zeros(n)
    momentum[rp-1:] = (velocity[rp-1:] * volumes[rp-1:]) / sigma

    # ── 4. Forces ──
    GRAVITY_THETA = 0.5
    SIGMA_ROCHE = 2.0
    REPULSION_EPSILON = 0.01
    REPULSION_FORCE_CAP = 100.0

    F_gravity = -GRAVITY_THETA * (z_scores * full_sigma)

    upper_sing = full_center + SIGMA_ROCHE * full_sigma
    lower_sing = full_center - SIGMA_ROCHE * full_sigma

    dist_upper = np.abs(prices - upper_sing) / full_sigma
    dist_lower = np.abs(prices - lower_sing) / full_sigma

    F_upper = np.where(z_scores > 0, 1.0 / (dist_upper**3 + REPULSION_EPSILON), 0.0)
    F_upper = np.clip(F_upper, 0, REPULSION_FORCE_CAP)

    F_lower = np.where(z_scores < 0, 1.0 / (dist_lower**3 + REPULSION_EPSILON), 0.0)
    F_lower = np.clip(F_lower, 0, REPULSION_FORCE_CAP)

    repulsion = np.where(z_scores > 0, -F_upper, F_lower)
    F_net = F_gravity + momentum + repulsion

    # ── 5. Wave Function ──
    E0 = -(z_scores ** 2) / 2.0
    E1 = -((z_scores - 2.0) ** 2) / 2.0
    E2 = -((z_scores + 2.0) ** 2) / 2.0

    max_E = np.maximum(np.maximum(E0, E1), E2)
    p0 = np.exp(E0 - max_E)
    p1 = np.exp(E1 - max_E)
    p2 = np.exp(E2 - max_E)
    total_p = p0 + p1 + p2
    p0 /= total_p
    p1 /= total_p
    p2 /= total_p

    eps = 1e-10
    entropy = -(p0 * np.log(p0 + eps) + p1 * np.log(p1 + eps) + p2 * np.log(p2 + eps))
    coherence = entropy / 1.09861228867  # ln(3)

    # ── 6. Archetype Detection ──
    VELOCITY_THRESHOLD = 0.5
    MOMENTUM_THRESHOLD = 5.0
    COHERENCE_THRESHOLD = 0.3

    roche_snap = (np.abs(z_scores) > 2.0) & (np.abs(velocity) > VELOCITY_THRESHOLD)
    structural_drive = (np.abs(momentum) > MOMENTUM_THRESHOLD) & (coherence < COHERENCE_THRESHOLD)

    # ── 7. Pattern Detection ──
    if 'high' in day_data.columns:
        highs = day_data['high'].values.astype(np.float64)
        lows = day_data['low'].values.astype(np.float64)
        opens = day_data['open'].values.astype(np.float64)
    else:
        highs = prices
        lows = prices
        opens = prices

    pattern_types, candlestick_types = self._detect_patterns_unified(opens, highs, lows, prices)

    # ── 8. Timestamps ──
    timestamps = np.zeros(n, dtype=np.float64)
    if 'timestamp' in day_data.columns:
        ts_col = day_data['timestamp']
        if pd.api.types.is_datetime64_any_dtype(ts_col):
            timestamps = ts_col.astype('int64').values / 1e9
        else:
            timestamps = ts_col.values.astype(np.float64)
    elif isinstance(day_data.index, pd.DatetimeIndex):
        timestamps = day_data.index.astype('int64').values / 1e9

    # ── 9. Assemble Results ──
    results = []
    for i in range(rp, n):
        z = z_scores[i]
        abs_z = abs(z)
        if abs_z < 1.0:
            lz = 'L1_STABLE'
        elif abs_z < 2.0:
            lz = 'CHAOS'
        elif z >= 2.0:
            lz = 'L2_ROCHE'
        else:
            lz = 'L3_ROCHE'

        a0 = math.sqrt(p0[i])
        a1 = math.sqrt(p1[i])
        a2 = math.sqrt(p2[i])

        slope_strength = (abs(full_slope[i]) * rp) / (full_sigma[i] + 1e-6)
        if slope_strength > 1.0:
            trend_direction = 'UP' if full_slope[i] > 0 else 'DOWN'
        else:
            trend_direction = 'RANGE'

        state = ThreeBodyQuantumState(
            center_position=full_center[i],
            upper_singularity=full_center[i] + 2.0 * full_sigma[i],
            lower_singularity=full_center[i] - 2.0 * full_sigma[i],
            event_horizon_upper=full_center[i] + 3.0 * full_sigma[i],
            event_horizon_lower=full_center[i] - 3.0 * full_sigma[i],
            particle_position=prices[i],
            particle_velocity=velocity[i],
            z_score=z,
            F_reversion=-0.5 * z * full_sigma[i],
            F_upper_repulsion=0.0,
            F_lower_repulsion=0.0,
            F_momentum=momentum[i],
            F_net=F_net[i],
            amplitude_center=a0,
            amplitude_upper=a1,
            amplitude_lower=a2,
            P_at_center=p0[i],
            P_near_upper=p1[i],
            P_near_lower=p2[i],
            entropy=entropy[i],
            coherence=coherence[i],
            pattern_maturity=0.0,
            momentum_strength=momentum[i],
            structure_confirmed=bool(structural_drive[i]),
            cascade_detected=bool(roche_snap[i]),
            spin_inverted=False,
            lagrange_zone=lz,
            stability_index=1.0,
            tunnel_probability=0.0, escape_probability=0.0,
            barrier_height=0.0,
            pattern_type=str(pattern_types[i]),
            candlestick_pattern=str(candlestick_types[i]),
            trend_direction_15m=trend_direction,
            hurst_exponent=0.5,
            adx_strength=0.0, dmi_plus=0.0, dmi_minus=0.0,
            sigma_fractal=full_sigma[i],
            term_pid=0.0,
            lyapunov_exponent=0.0,
            market_regime='STABLE',
            timestamp=timestamps[i]
        )

        results.append({
            'bar_idx': i,
            'state': state,
            'price': prices[i],
            'structure_ok': lz in ('L2_ROCHE', 'L3_ROCHE')
        })

    return results
```

### Step 3: Update `batch_compute_states()` to dispatch

Add a dispatch check at the top of `batch_compute_states()`, right after the `n < rp` check (after line 195):

```python
    if n < rp:
        return []

    # Dispatch: GPU or CPU
    if not self.use_gpu:
        return self._batch_compute_cpu(day_data, params)

    # ... rest of existing GPU code unchanged ...
```

---

## Verification

### Test 1: CPU fallback works
```bash
python -c "
import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'  # Force CPU mode
from core.quantum_field_engine import QuantumFieldEngine
import pandas as pd, numpy as np

# Create synthetic data
n = 100
df = pd.DataFrame({
    'price': np.cumsum(np.random.randn(n)) + 20000,
    'volume': np.random.randint(1, 100, n).astype(float),
    'timestamp': np.arange(n, dtype=float) * 15.0
})
engine = QuantumFieldEngine()
results = engine.batch_compute_states(df)
print(f'CPU fallback: {len(results)} states computed')
print(f'First z_score: {results[0][\"state\"].z_score:.4f}')
print('PASS')
"
```

### Test 2: GPU path still works (unchanged)
```bash
python -c "
from core.quantum_field_engine import QuantumFieldEngine
import pandas as pd, numpy as np

n = 1000
df = pd.DataFrame({
    'price': np.cumsum(np.random.randn(n)) + 20000,
    'volume': np.random.randint(1, 100, n).astype(float),
    'timestamp': np.arange(n, dtype=float) * 15.0
})
engine = QuantumFieldEngine()
print(f'GPU mode: {engine.use_gpu}')
results = engine.batch_compute_states(df)
print(f'GPU path: {len(results)} states computed')
print('PASS')
"
```

### Test 3: Full pipeline still works
```bash
python training/orchestrator.py --no-dashboard --iterations 50
```
(No `--fresh` — uses existing discovery checkpoint from previous run)

---

## File Summary

| File | Action |
|------|--------|
| `core/quantum_field_engine.py` | Graceful CUDA init + `_batch_compute_cpu()` method + dispatch in `batch_compute_states()` |

## Key Design Decisions
- **`sliding_window_view`** — zero-copy rolling windows, O(1) memory overhead
- **Coincident windows** — `windows[i]` = `prices[i:i+rp]`, result maps to bar `i + rp - 1` (ending at current bar, not predictive)
- **Exact physics replication** — every constant, threshold, and formula matches `cuda_physics.py`
- **No new dependencies** — uses numpy builtins only
- **GPU path untouched** — existing CUDA code remains the primary path, CPU is fallback only
- **Graceful degradation** — `__init__` no longer crashes without CUDA

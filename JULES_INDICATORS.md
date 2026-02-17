# Jules Task: Compute ADX/DMI and Hurst Exponent in GPU Kernel

## Problem

`ThreeBodyQuantumState` has fields for `adx_strength`, `dmi_plus`, `dmi_minus`, and `hurst_exponent` — but they're all hardcoded to 0.0. Without these, the system cannot distinguish:

- **Trending markets** (ADX>25, Hurst>0.5) — momentum strategies work
- **Brownian noise** (ADX<15, Hurst≈0.5) — random walk, no edge exists
- **Mean-reverting regimes** (Hurst<0.4) — fade moves, don't ride them
- **Directional bias** (DMI+ vs DMI-) — confirms long vs short conviction

These indicators are critical for the star schema: a Roche Snap in a trending market is a completely different pattern than one in Brownian noise.

## Solution

Add two new CUDA kernels to `core/cuda_physics.py`:
1. `compute_adx_dmi_kernel` — Wilder's ADX with DMI+/-
2. `compute_hurst_kernel` — Rescaled Range (R/S) Hurst exponent

Wire them into `batch_compute_states()` in `core/quantum_field_engine.py`.

---

## File: `core/cuda_physics.py`

### Constants to add at top

```python
ADX_PERIOD = 14
HURST_WINDOW = 100
HURST_MIN_WINDOW = 30  # Minimum bars before computing Hurst
```

### Kernel 1: ADX/DMI

ADX requires sequential smoothing (Wilder's EMA). Since CUDA threads are per-bar, use a **two-pass approach**:
- Pass 1: Compute raw True Range (TR), +DM, -DM per bar (fully parallel)
- Pass 2: Sequential smoothing on CPU (only 14-period EMA, cheap)

**Add this kernel for Pass 1 (parallel per bar):**

```python
@cuda.jit
def compute_dm_tr_kernel(highs, lows, closes,
                          out_tr, out_plus_dm, out_minus_dm):
    """
    Pass 1: Compute raw True Range and Directional Movement per bar.
    Fully parallel — one thread per bar.
    """
    i = cuda.grid(1)
    n = highs.shape[0]

    if i < n:
        if i == 0:
            out_tr[i] = highs[i] - lows[i]
            out_plus_dm[i] = 0.0
            out_minus_dm[i] = 0.0
        else:
            # True Range = max(H-L, |H-prevC|, |L-prevC|)
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            out_tr[i] = max(hl, max(hc, lc))

            # +DM and -DM
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                out_plus_dm[i] = up_move
            else:
                out_plus_dm[i] = 0.0

            if down_move > up_move and down_move > 0:
                out_minus_dm[i] = down_move
            else:
                out_minus_dm[i] = 0.0
```

**Pass 2 — Wilder smoothing on CPU** (add as a regular Python function, NOT a kernel):

```python
def compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, period=14):
    """
    Pass 2: Wilder's smoothed ADX/DMI computation.
    Sequential but fast (single pass over arrays).

    Returns: (adx, dmi_plus, dmi_minus) — all numpy arrays length n
    """
    n = len(tr_raw)
    adx = np.zeros(n)
    dmi_plus = np.zeros(n)
    dmi_minus = np.zeros(n)

    if n < period + 1:
        return adx, dmi_plus, dmi_minus

    # Initial sums (first `period` bars)
    smooth_tr = np.sum(tr_raw[1:period+1])
    smooth_plus = np.sum(plus_dm_raw[1:period+1])
    smooth_minus = np.sum(minus_dm_raw[1:period+1])

    # First DI values
    if smooth_tr > 0:
        dmi_plus[period] = 100.0 * smooth_plus / smooth_tr
        dmi_minus[period] = 100.0 * smooth_minus / smooth_tr

    # First DX
    di_sum = dmi_plus[period] + dmi_minus[period]
    if di_sum > 0:
        dx_first = 100.0 * abs(dmi_plus[period] - dmi_minus[period]) / di_sum
    else:
        dx_first = 0.0

    # Wilder smoothing for remaining bars
    dx_sum = dx_first
    dx_count = 1

    for i in range(period + 1, n):
        # Wilder smoothing: smooth = prev_smooth - (prev_smooth / period) + current
        smooth_tr = smooth_tr - (smooth_tr / period) + tr_raw[i]
        smooth_plus = smooth_plus - (smooth_plus / period) + plus_dm_raw[i]
        smooth_minus = smooth_minus - (smooth_minus / period) + minus_dm_raw[i]

        if smooth_tr > 0:
            dmi_plus[i] = 100.0 * smooth_plus / smooth_tr
            dmi_minus[i] = 100.0 * smooth_minus / smooth_tr

        di_sum = dmi_plus[i] + dmi_minus[i]
        if di_sum > 0:
            dx = 100.0 * abs(dmi_plus[i] - dmi_minus[i]) / di_sum
        else:
            dx = 0.0

        # ADX = Wilder smoothed DX
        if dx_count < period:
            dx_sum += dx
            dx_count += 1
            if dx_count == period:
                adx[i] = dx_sum / period
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx) / period

    return adx, dmi_plus, dmi_minus
```

### Kernel 2: Hurst Exponent (R/S Method)

Hurst measures if price is trending (H>0.5), random walk (H≈0.5), or mean-reverting (H<0.5).

**Approach**: Rescaled Range (R/S) analysis over rolling windows. Each thread computes Hurst for one bar using the preceding `HURST_WINDOW` bars.

```python
@cuda.jit
def compute_hurst_kernel(prices, out_hurst, window_size):
    """
    Rescaled Range (R/S) Hurst exponent per bar.
    Uses 4 sub-window sizes: window/8, window/4, window/2, window.
    Linear regression of log(R/S) vs log(n) gives Hurst.
    """
    i = cuda.grid(1)
    n = prices.shape[0]

    if i < n:
        out_hurst[i] = 0.5  # Default: Brownian

        if i < window_size:
            return

        # Sub-window sizes for R/S regression
        # We'll use 4 sizes: w/8, w/4, w/2, w
        sizes = cuda.local.array(4, dtype=numba.int32)
        sizes[0] = max(window_size // 8, 4)
        sizes[1] = max(window_size // 4, 8)
        sizes[2] = max(window_size // 2, 16)
        sizes[3] = window_size

        log_n = cuda.local.array(4, dtype=numba.float64)
        log_rs = cuda.local.array(4, dtype=numba.float64)

        for s_idx in range(4):
            sz = sizes[s_idx]
            start = i - sz + 1

            # Compute returns within sub-window
            mean_ret = 0.0
            for k in range(start + 1, i + 1):
                mean_ret += (prices[k] - prices[k-1])
            mean_ret /= (sz - 1)

            # Cumulative deviation from mean
            cum_dev = 0.0
            max_dev = -1e30
            min_dev = 1e30
            std_sum = 0.0

            for k in range(start + 1, i + 1):
                ret = (prices[k] - prices[k-1]) - mean_ret
                cum_dev += ret
                if cum_dev > max_dev:
                    max_dev = cum_dev
                if cum_dev < min_dev:
                    min_dev = cum_dev
                std_sum += ret * ret

            R = max_dev - min_dev
            S = math.sqrt(std_sum / (sz - 1)) if sz > 1 else 1e-10
            S = max(S, 1e-10)

            rs = R / S
            log_n[s_idx] = math.log(float(sz))
            log_rs[s_idx] = math.log(max(rs, 1e-10))

        # Linear regression: log(R/S) = H * log(n) + c
        # Hurst = slope
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        for j in range(4):
            sum_x += log_n[j]
            sum_y += log_rs[j]
            sum_xy += log_n[j] * log_rs[j]
            sum_xx += log_n[j] * log_n[j]

        denom = 4.0 * sum_xx - sum_x * sum_x
        if abs(denom) > 1e-12:
            hurst = (4.0 * sum_xy - sum_x * sum_y) / denom
        else:
            hurst = 0.5

        # Clamp to [0, 1]
        out_hurst[i] = max(0.0, min(1.0, hurst))
```

**IMPORTANT**: Add `import numba` at the top of `cuda_physics.py` (needed for `numba.int32`, `numba.float64` in `cuda.local.array`).

---

## File: `core/quantum_field_engine.py`

### Wire new kernels into `batch_compute_states()`

After the existing `compute_physics_kernel` and `detect_archetype_kernel` calls, add:

```python
# ── ADX/DMI (Pass 1: GPU, Pass 2: CPU) ──
from core.cuda_physics import compute_dm_tr_kernel, compute_adx_dmi_cpu, compute_hurst_kernel

# Need high, low, close arrays
if 'high' in day_data.columns:
    d_highs = cuda.to_device(day_data['high'].values.astype(np.float64))
    d_lows = cuda.to_device(day_data['low'].values.astype(np.float64))
    d_closes = cuda.to_device(prices)  # already on device or use close column
else:
    d_highs = d_prices  # fallback to price
    d_lows = d_prices
    d_closes = d_prices

d_tr = cuda.device_array(n, dtype=np.float64)
d_plus_dm = cuda.device_array(n, dtype=np.float64)
d_minus_dm = cuda.device_array(n, dtype=np.float64)

compute_dm_tr_kernel[blocks, threads](d_highs, d_lows, d_closes,
                                       d_tr, d_plus_dm, d_minus_dm)
cuda.synchronize()

# Pass 2 on CPU (sequential Wilder smoothing — fast for n=5000)
tr_raw = d_tr.copy_to_host()
plus_dm_raw = d_plus_dm.copy_to_host()
minus_dm_raw = d_minus_dm.copy_to_host()
adx_arr, dmi_plus_arr, dmi_minus_arr = compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw)

# ── Hurst Exponent (GPU) ──
d_hurst = cuda.device_array(n, dtype=np.float64)
HURST_WINDOW = 100

compute_hurst_kernel[blocks, threads](d_prices, d_hurst, HURST_WINDOW)
cuda.synchronize()

hurst_arr = d_hurst.copy_to_host()
```

### Update ThreeBodyQuantumState construction

In the result assembly loop where `ThreeBodyQuantumState` is created, replace the hardcoded values:

```python
# Old:
hurst_exponent=0.5,
adx_strength=0.0, dmi_plus=0.0, dmi_minus=0.0,

# New:
hurst_exponent=hurst_arr[i],
adx_strength=adx_arr[i], dmi_plus=dmi_plus_arr[i], dmi_minus=dmi_minus_arr[i],
```

### Also update `_batch_compute_cpu()` (CPU fallback)

If the CPU fallback method exists (from JULES_CPU_PHYSICS.md), add equivalent numpy-based ADX/DMI and Hurst computation there. The `compute_adx_dmi_cpu()` function already works on numpy arrays. For Hurst, implement a numpy version:

```python
def _compute_hurst_numpy(prices, window=100):
    """Numpy Hurst exponent via R/S method."""
    n = len(prices)
    hurst = np.full(n, 0.5)

    for i in range(window, n):
        log_ns = []
        log_rs_vals = []

        for sz in [window//8, window//4, window//2, window]:
            sz = max(sz, 4)
            segment = prices[i-sz+1:i+1]
            returns = np.diff(segment)
            if len(returns) < 2:
                continue

            mean_r = returns.mean()
            devs = np.cumsum(returns - mean_r)
            R = devs.max() - devs.min()
            S = max(returns.std(ddof=1), 1e-10)

            log_ns.append(np.log(sz))
            log_rs_vals.append(np.log(max(R/S, 1e-10)))

        if len(log_ns) >= 2:
            slope, _, _, _, _ = np.polyfit(log_ns, log_rs_vals, 1, full=False, cov=False)
            # np.polyfit returns coefficients, slope is first
            coeffs = np.polyfit(log_ns, log_rs_vals, 1)
            hurst[i] = np.clip(coeffs[0], 0.0, 1.0)

    return hurst
```

---

## File: `training/fractal_clustering.py`

### Expand feature vector to include market regime

Update the feature extraction in `create_templates()` to include ADX, Hurst, and DMI differential:

```python
# Current 7D (becomes 7D base):
base = [abs(z), abs(v), abs(m), c, tf_scale, depth, parent_ctx]

# NEW: Add 3 market regime features (total: 10D before star schema, 14D after)
adx = getattr(p.state, 'adx_strength', 0.0) if hasattr(p, 'state') else 0.0
hurst = getattr(p.state, 'hurst_exponent', 0.5) if hasattr(p, 'state') else 0.5
dmi_diff = 0.0
if hasattr(p, 'state'):
    dmi_diff = getattr(p.state, 'dmi_plus', 0.0) - getattr(p.state, 'dmi_minus', 0.0)

features.append(base + [adx / 100.0, hurst, dmi_diff / 100.0])
```

This normalizes ADX (0-100) and DMI diff to similar scales as other features.

**IMPORTANT**: Also update `refine_clusters()` to use the same extended vector when computing sub-template centroids.

---

## File: `training/fractal_discovery_agent.py`

### Update PatternEvent to store regime indicators

The state already contains ADX/Hurst after the kernel update. No changes needed to PatternEvent itself since it stores the full `ThreeBodyQuantumState` in `state` field. The clustering reads from `p.state.adx_strength` etc.

---

## Verification

### Test 1: Kernel outputs

```bash
python -c "
from core.quantum_field_engine import QuantumFieldEngine
import pandas as pd, numpy as np

n = 500
df = pd.DataFrame({
    'price': np.cumsum(np.random.randn(n) * 0.5) + 20000,
    'high': np.cumsum(np.random.randn(n) * 0.5) + 20001,
    'low': np.cumsum(np.random.randn(n) * 0.5) + 19999,
    'close': np.cumsum(np.random.randn(n) * 0.5) + 20000,
    'open': np.cumsum(np.random.randn(n) * 0.5) + 20000,
    'volume': np.random.randint(1, 100, n).astype(float),
    'timestamp': np.arange(n, dtype=float) * 15.0
})
engine = QuantumFieldEngine()
results = engine.batch_compute_states(df)
if results:
    s = results[-1]['state']
    print(f'ADX: {s.adx_strength:.1f}')
    print(f'DMI+: {s.dmi_plus:.1f}  DMI-: {s.dmi_minus:.1f}')
    print(f'Hurst: {s.hurst_exponent:.3f}')
    assert s.adx_strength >= 0, 'ADX should be non-negative'
    assert 0 <= s.hurst_exponent <= 1, 'Hurst should be [0,1]'
    print('PASS')
"
```

### Test 2: Full pipeline with new features

```bash
python training/orchestrator.py --fresh --no-dashboard --iterations 50
```

Expected: Pipeline runs end-to-end. Clustering now uses 10D (or 14D with star schema) features. Template count may change slightly due to richer feature space.

### Test 3: Verify Brownian detection

```python
# Random walk should give Hurst ≈ 0.5
prices = np.cumsum(np.random.randn(1000))  # Pure Brownian
# Trending series should give Hurst > 0.6
trending = np.cumsum(np.random.randn(1000) + 0.05)  # Drift
# Mean-reverting should give Hurst < 0.4
mean_rev = np.zeros(1000)
for i in range(1, 1000):
    mean_rev[i] = mean_rev[i-1] * 0.95 + np.random.randn()
```

---

## File Summary

| File | Action |
|------|--------|
| `core/cuda_physics.py` | Add `compute_dm_tr_kernel` (GPU), `compute_adx_dmi_cpu` (CPU), `compute_hurst_kernel` (GPU) |
| `core/quantum_field_engine.py` | Wire new kernels into `batch_compute_states()`, update state construction |
| `training/fractal_clustering.py` | Expand feature vector from 7D to 10D (add ADX, Hurst, DMI diff) |

## Key Design Decisions

- **ADX hybrid approach**: TR/DM computed on GPU (parallel), Wilder smoothing on CPU (sequential but only ~5000 bars, <1ms)
- **Hurst via R/S method**: 4 sub-window sizes for regression stability. Each thread independent.
- **Feature normalization**: ADX/100, Hurst raw [0,1], DMI_diff/100 — keeps features in similar scale for clustering
- **Hurst window = 100 bars**: At 15s bars = 25 minutes lookback. Sufficient for regime detection.
- **No new dependencies**: Uses existing numba.cuda + numpy

## Why This Matters

The clustering will now naturally separate:
- **High-ADX Roche Snaps** (trending breakout) from **Low-ADX Roche Snaps** (noise)
- **H>0.5 Structural Drives** (persistent momentum) from **H<0.5** (mean-reverting fakeout)
- **DMI+ > DMI- with Roche** (strong long setup) from **DMI- > DMI+** (strong short setup)

This directly feeds into the star schema: the parent chain's ADX/Hurst at each level tells you if the macro structure supports the micro entry.

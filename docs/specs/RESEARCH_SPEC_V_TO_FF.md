# Async Research Spec: Analyses V–FF

## For: Claude Code (VS Code) — run independently, no coordination needed

---

## Design Principles

1. **Each analysis is a standalone script.** No dependency between V, W, X, Y, Z, AA-FF.
   Run any subset in any order. Each reads from ATLAS parquet or trade logs and writes
   to its own output directory.

2. **No core/ or live/ changes.** Pure research. Import from `core/` and
   `tools/research/` for physics + data loading. Write results to
   `reports/research/` and plots to `tools/plots/research/`.

3. **Reuse existing infrastructure.** `tools/research/data.py` loads ATLAS
   and computes 192D physics matrices. `tools/research/imr.py` does regime
   detection. Don't rebuild these — import them.

4. **Each script has a GATE.** A quantified threshold that determines
   "promote to pipeline" vs "kill and move on." No ambiguous conclusions.

5. **Journal continuity.** Each script appends its results to
   `docs/reference/RESEARCH_JOURNAL.txt` in the established format.

---

## File Structure

```
scripts/research/
├── analysis_v_trajectory_knn.py     # 8-point trajectory extrapolation
├── analysis_w_partial_bar_u.py      # Analysis U under live conditions
├── analysis_x_counter_trend.py      # Counter-trend scale analysis
├── analysis_y_regime_knn.py         # Regime-conditional k-NN
├── analysis_z_fractal_dims.py       # Shi 2018 fractal dimension features
├── analysis_aa_seed_shape.py        # Seed library shape → direction signal
├── analysis_bb_stacked_direction.py # 176D multi-TF → GBM direction model
├── analysis_cc_epnl_prediction.py   # E[PnL] tick prediction & calibration
├── analysis_dd_level_proximity.py   # Auto-detected levels → trade filter
├── analysis_ee_stop_loss_opt.py     # SL width sweep + physics-conditional SL
├── analysis_ff_conviction_audit.py  # Conviction calibration + recalibration
└── README.md                        # This file (abridged)

reports/research/
├── V_trajectory_knn/                # Per-analysis output dirs
│   ├── results.txt
│   └── *.png
├── W_partial_bar_u/
├── X_counter_trend/
├── Y_regime_knn/
├── Z_fractal_dims/
├── AA_seed_shape/
├── BB_stacked_direction/
├── CC_epnl_prediction/
├── DD_level_proximity/
├── EE_stop_loss_opt/
└── FF_conviction_audit/
```

---

## Shared Boilerplate (copy into each script)

```python
#!/usr/bin/env python3
"""
Analysis [LETTER]: [TITLE]
===========================
Standalone research script. No core/ or live/ modifications.

Usage:
    python scripts/research/analysis_[x]_[name].py
    python scripts/research/analysis_[x]_[name].py --data DATA/ATLAS_1MONTH
    python scripts/research/analysis_[x]_[name].py --data DATA/ATLAS --analysis-days 270

Outputs:
    reports/research/[LETTER]_[name]/results.txt
    reports/research/[LETTER]_[name]/*.png
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.research.data import (
    load_atlas_tf, compute_tf_physics, extract_16d,
    build_stacked_matrices, TF_HIERARCHY, TF_SECONDS, FEATURE_NAMES,
)
from tools.research.imr import compute_price_imr, detect_regimes, compute_regime_oracle

# Output directory
ANALYSIS_ID = '[LETTER]_[name]'
OUT_DIR = os.path.join('reports', 'research', ANALYSIS_ID)
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description=f'Analysis {ANALYSIS_ID}')
    parser.add_argument('--data', default='DATA/ATLAS_1MONTH',
                        help='ATLAS data directory')
    parser.add_argument('--base-tf', default='15m',
                        help='Base timeframe (default: 15m)')
    parser.add_argument('--context-days', type=int, default=21,
                        help='Warmup/context days')
    parser.add_argument('--analysis-days', type=int, default=0,
                        help='Analysis window days (0=all remaining)')
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f'Analysis {ANALYSIS_ID}')
    print(f'Data: {args.data}, Base TF: {args.base_tf}')
    print('=' * 60)

    # --- Load data ---
    # [analysis-specific code here]

    # --- Write results ---
    elapsed = time.perf_counter() - t0
    print(f'\nCompleted in {elapsed:.1f}s')
    print(f'Results: {OUT_DIR}/results.txt')

    # --- Append to journal ---
    _append_journal(ANALYSIS_ID, results_text)


def _append_journal(analysis_id: str, text: str):
    """Append results to the waveform analysis journal."""
    journal = 'docs/reference/RESEARCH_JOURNAL.txt'
    with open(journal, 'a', encoding='utf-8') as f:
        f.write(f'\n\n{"=" * 77}\n')
        f.write(f'ANALYSIS {analysis_id.upper()} (auto-generated {time.strftime("%Y-%m-%d")})\n')
        f.write(f'{"=" * 77}\n\n')
        f.write(text)


if __name__ == '__main__':
    main()
```

---

## Analysis V: Trajectory k-NN Extrapolation

### Why This Is #1

The journal's final insight (2026-03-08): 8 sequential 192D snapshots can
extrapolate the 9th. Analysis P proved transitions are deterministic (100%
at scale). Analysis U proved k-NN on single-point 192D gives 89.3% direction.
Trajectory k-NN combines both — more context for the same proven method.

### Script: `scripts/research/analysis_v_trajectory_knn.py`

### Method

```python
# 1. Build 192D state matrix (same as existing analyses)
sample_ts, X = build_stacked_matrices(base_df, tf_data, context_days, ...)

# 2. Build trajectory windows: 8 consecutive 192D states → 1 target
WINDOW = 8
trajectories = []  # each = (8 × 192D flattened to 1536D)
targets = []       # signed MFE of bar after window

for i in range(WINDOW, len(X)):
    traj = X[i - WINDOW : i].flatten()  # 1536D
    trajectories.append(traj)
    # Target: signed MFE from bar i
    targets.append(oracle_signed_mfe[i])

X_traj = np.array(trajectories)
y_traj = np.array(targets)

# 3. Train/test split (chronological — no shuffle)
split = int(len(X_traj) * 0.75)
X_train, X_test = X_traj[:split], X_traj[split:]
y_train, y_test = y_traj[:split], y_traj[split:]

# 4. Scale (fit on train only)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

# 5. k-NN (same as Analysis U, but on 1536D trajectories)
from sklearn.neighbors import NearestNeighbors
k = min(50, len(X_train) // 10)
knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
knn.fit(X_train_sc)

# 6. For each test point: find k neighbors, build CI
dists, indices = knn.kneighbors(X_test_sc)
results = []
for i in range(len(X_test)):
    nbr_idx = indices[i, 1:]  # exclude self
    nbr_mfe = y_train[nbr_idx]
    
    p10, p25, p50, p75, p90 = np.percentile(nbr_mfe, [10, 25, 50, 75, 90])
    predicted_dir = 'LONG' if p50 > 0 else 'SHORT'
    actual_dir = 'LONG' if y_test[i] > 0 else 'SHORT'
    
    # Neighbor consensus
    nbr_long_pct = np.mean(nbr_mfe > 0)
    consensus = max(nbr_long_pct, 1 - nbr_long_pct)
    
    results.append({
        'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
        'predicted_dir': predicted_dir, 'actual_dir': actual_dir,
        'actual_signed_mfe': y_test[i],
        'consensus': consensus,
        'ci_width': p75 - p25,
    })

# 7. Metrics (same as Analysis U for direct comparison)
dir_correct = sum(1 for r in results if r['predicted_dir'] == r['actual_dir'])
dir_accuracy = dir_correct / len(results)
baseline = max(sum(1 for r in results if r['actual_dir'] == 'LONG'),
               sum(1 for r in results if r['actual_dir'] == 'SHORT')) / len(results)
lift = dir_accuracy - baseline

ci50_coverage = sum(1 for r in results 
    if r['p25'] <= r['actual_signed_mfe'] <= r['p75']) / len(results)
ci80_coverage = sum(1 for r in results 
    if r['p10'] <= r['actual_signed_mfe'] <= r['p90']) / len(results)
```

### Comparison Against Analysis U (Single-Point Baseline)

Also run standard Analysis U (single 192D point, same k, same data split)
within this script to produce an apples-to-apples comparison:

```python
# Single-point baseline (Analysis U equivalent)
knn_single = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
knn_single.fit(X_train_single_sc)  # 192D, not 1536D
# ... same CI computation ...
```

### Output Table (report must include this)

```
                        Single-Point (U)    Trajectory (V)    Delta
Direction accuracy      ?.?%                ?.?%              +?.?%
Lift over baseline      +?.?%               +?.?%             +?.?%
50% CI coverage         ?.?%                ?.?%
80% CI coverage         ?.?%                ?.?%
CI width (median)       ?.? ticks           ?.? ticks
Consensus 90%+ acc      ?.?% (N=?)          ?.?% (N=?)
```

### Plots

1. **V vs U direction accuracy by consensus bin** (side-by-side bars)
2. **CI calibration: V vs U** (predicted CI vs actual hit rate)
3. **P50 vs actual scatter** (V trajectory, with U overlaid in gray)
4. **Sample trajectories** (5 best predictions, 5 worst, 8-bar 192D paths)

### Gate

```
PROMOTE if: V direction accuracy > U direction accuracy by >= 2 percentage points
            AND CI coverage is within 5pp of U (not degraded)
            AND consensus 90%+ bin has >= 95% accuracy with N >= 100

KILL if:    V direction accuracy <= U accuracy (trajectory adds nothing)
            OR CI coverage degrades by > 10pp (overfitting)

DEFER if:   V > U but consensus bin has < 100 samples (need more data)
```

### Estimated Runtime

- 1-month data (~1,800 bars, ~1,800 trajectories): ~30 seconds
- Full ATLAS (~17,000 bars, ~17,000 trajectories): ~5 minutes
- Bottleneck: k-NN on 1536D is O(N² × D) — sklearn uses ball tree

### If V Passes — Next Steps (not in this script)

- Integration spec: replace brain's template_id key with trajectory hash
- Or: add trajectory k-NN as a confidence overlay on existing pipeline
- Either way: observation-only first (log prediction, don't gate on it)

### Variants to Test (within the same script)

```python
# Variant A: Flattened trajectory (8 × 192D = 1536D) — default
# Variant B: Delta trajectory (7 × 192D differences between consecutive snapshots = 1344D)
# Variant C: Summary trajectory (mean, std, slope of each of 192 features across 8 bars = 576D)
# Variant D: Window sizes 4, 8, 12, 16 (how much history matters?)

for window in [4, 8, 12, 16]:
    for variant in ['flat', 'delta', 'summary']:
        run_experiment(window, variant)
        # Log: window, variant, direction_acc, ci_coverage, n_test
```

---

## Analysis W: Analysis U Under Live Conditions (Partial Bar Robustness)

### Why

Analysis T showed -5.6% direction degradation and -10% when 1D is proxied.
But T tested the signed MFE OLS model (Analysis L). Nobody tested whether
Analysis U's k-NN survives partial bars. U is the production candidate —
we need to know if stale slow-TF bars kill it.

### Script: `scripts/research/analysis_w_partial_bar_u.py`

### Method

```python
# 1. Build COMPLETE 192D matrix (ground truth)
sample_ts_complete, X_complete = build_stacked_matrices(
    base_df, tf_data, context_days, ...)

# 2. Build PARTIAL 192D matrix (simulate live conditions)
# For each slow TF (4h, 1D, 1W): substitute with previous completed bar
# This means: at bar t, the 4h features come from the 4h bar that ENDED
# before t (not the one currently forming).
#
# Analysis T's method: use N-2 instead of N-1 for slow TFs.
# Even more aggressive: for 1D, use the PREVIOUS DAY's bar (always stale by 0-23h)
# For 1W, use PREVIOUS WEEK's bar (always stale by 0-5 days)

sample_ts_partial, X_partial = build_stacked_matrices(
    base_df, tf_data, context_days, ...,
    partial_bar_mode=True  # flag to use N-2 for slow TFs
)

# 3. Run Analysis U's k-NN on BOTH matrices (same oracle targets)
results_complete = run_knn_ci(X_complete, oracle_signed_mfe, k=50)
results_partial = run_knn_ci(X_partial, oracle_signed_mfe, k=50)

# 4. Also run with selective degradation (one TF at a time)
for degrade_tf in ['4h', '1D', '1W']:
    X_degrade = build_stacked_matrices(..., degrade_tfs=[degrade_tf])
    results_degrade = run_knn_ci(X_degrade, oracle_signed_mfe, k=50)
    # This tells us WHICH slow TF hurts most
```

### Output Table

```
Scenario              Dir Acc   CI 50%   CI 80%   vs Complete
──────────────────────────────────────────────────────────────
Complete bars          ?.?%     ?.?%     ?.?%     baseline
Partial (all slow)     ?.?%     ?.?%     ?.?%     -?.?%
Degrade 4h only        ?.?%     ?.?%     ?.?%     -?.?%
Degrade 1D only        ?.?%     ?.?%     ?.?%     -?.?%
Degrade 1W only        ?.?%     ?.?%     ?.?%     -?.?%
```

### Plots

1. **Direction accuracy: complete vs partial** (bar chart, 5 scenarios)
2. **CI calibration: complete vs partial** (overlay on same axes)
3. **Per-TF degradation waterfall** (which TF hurts most)
4. **Accuracy by consensus bin: complete vs partial** (side-by-side)

### Gate

```
PROMOTE if: Partial-bar direction accuracy is within 5pp of complete-bar
            AND CI coverage within 5pp
            → U is live-ready, deploy into observation layer

KILL if:    Degradation > 15pp → U needs partial bar interpolation before live
            (escalates ROADMAP item #2 to high priority)

DEFER if:   One specific TF (e.g., 1W) causes >10pp drop
            → Can deploy U live WITHOUT that TF's features (zero them out)
```

### Build Note

`build_stacked_matrices()` in `tools/research/data.py` already handles TF
alignment with the N-1/N-2 fix (see journal "CRITICAL BUG" section). Adding
a `partial_bar_mode` or `degrade_tfs` parameter is a ~20 line change to
that function. The rest of the script is pure k-NN (copy from Analysis U).

---

## Analysis X: Counter-Trend Scale Analysis

### Why

The journal explicitly flagged this as a prerequisite for E[PnL] integration
(see "RESEARCH PREREQUISITES" section). Counter-trend scalps are $26K IS,
$5.8K OOS — real profit. But we don't know WHY they work. "Where in the
trend" and "exhaustion vs fighting fresh" have never been measured.

### Script: `scripts/research/analysis_x_counter_trend.py`

### Data Source

This script reads the **forward pass trade log**, not raw ATLAS data.
It needs oracle labels (direction, MFE, MAE) per trade.

```python
# Primary: OOS trade log from latest forward pass
df = pd.read_csv('reports/oos/oracle_trade_log.csv')

# Identify counter-trend trades
# oracle_label > 0 = oracle says LONG; direction == 'SHORT' = system went SHORT
# Counter-trend = system direction != oracle direction AND profitable
df['is_counter_trend'] = (
    ((df['oracle_label'] > 0) & (df['direction'] == 'SHORT')) |
    ((df['oracle_label'] < 0) & (df['direction'] == 'LONG'))
) & (df['pnl'] > 0)
```

### Metrics to Compute

For each counter-trend trade:

```python
# 1. Where in the trend (requires oracle data)
# oracle_mfe = total move the oracle predicted
# entry_offset = how far into the oracle move we entered
# (need entry price vs oracle entry price — may need signal_log)
'bars_into_trend': ...,     # how many bars since oracle trend started
'trend_pct_complete': ...,  # bars_into / oracle_hold_bars (0=start, 1=end)

# 2. Capture ratio
'capture_ratio': trade_pnl / (oracle_mfe * tick_value),  # how much of available move we got

# 3. Exhaustion signal
'wave_maturity_at_entry': ...,  # from TBN if available in trade log
'z_score_at_entry': ...,
'hurst_at_entry': ...,

# 4. Survival by depth
'entry_depth': ...,  # fractal depth level

# 5. TP ambition vs reality
'tp_ticks': ...,
'oracle_mfe_ticks': ...,
'ambition_ratio': tp_ticks / oracle_mfe_ticks,
```

### Core Hypothesis

```
H0: Counter-trend trades entered in the first 30% of a trend are
    as profitable as those entered in the last 30%.

H1: Counter-trend trades entered after 70% trend completion (exhaustion
    zone) have significantly higher WR and capture ratio.
```

Test with Welch t-test on PnL between early (trend_pct < 0.30) and
late (trend_pct > 0.70) counter-trend entries.

### Output Table

```
Trend Position     N     WR%    Avg PnL   Capture%   Avg Wave Maturity
──────────────────────────────────────────────────────────────────────
Early (0-30%)      ?     ?.?%   $?.??     ?.?%       ?.??
Mid (30-70%)       ?     ?.?%   $?.??     ?.?%       ?.??
Late (70-100%)     ?     ?.?%   $?.??     ?.?%       ?.??

By Depth:
Depth 3-5          ?     ?.?%   $?.??
Depth 6-8          ?     ?.?%   $?.??
Depth 9-12         ?     ?.?%   $?.??
```

### Plots

1. **PnL by trend position** (scatter: x=trend_pct_complete, y=pnl, colored by WR)
2. **Capture ratio distribution** (histogram, early vs late overlay)
3. **Wave maturity at entry** (box plot, winning vs losing counter-trends)
4. **Survival by depth** (WR bars per depth bucket)

### Gate

```
PROMOTE if: Late-trend (>70%) counter-trends have WR > 65% AND
            avg PnL > $15/trade AND N >= 50
            → Build exhaustion-aware E[PnL] model that allows late counter-trends

KILL if:    No significant difference between early and late
            → Counter-trend profit is random, not exhaustion-driven
            → Block all counter-trends in pipeline (saves simplicity)

DEFER if:   Significant but N < 50 in late bucket
            → Need more OOS data (run on IS for larger sample, validate on OOS)
```

### Data Dependency

Requires `oracle_trade_log.csv` with columns: `direction`, `oracle_label`,
`pnl`, `entry_depth`, `oracle_mfe`, `hold_bars`, `tp_ticks`, `wave_maturity`,
`z_score`, `hurst`. Some of these were added in Session 3 (2026-03-08) — verify
they exist. If missing, skip the metric and note it in results.

---

## Analysis Y: Regime-Conditional k-NN

### Why

Analysis E proved dp/dt grouping gives stable 2.1x direction R² lift by
keeping homogeneous bars together. Analysis U's k-NN finds neighbors globally.
Constraining k-NN to regime-local neighbors should find BETTER neighbors —
same market conditions, tighter CI.

### Script: `scripts/research/analysis_y_regime_knn.py`

### Method

```python
# 1. Build 192D matrix + regime labels
sample_ts, X = build_stacked_matrices(...)
regimes = detect_regimes(base_df, context_days)
regime_labels = assign_regime_to_bar(sample_ts, regimes)  # per-bar regime ID

# 2. For each test point: k-NN within SAME regime only
results_local = []
for i in test_indices:
    my_regime = regime_labels[i]
    
    # Find training points in same regime (or ±1 adjacent)
    regime_mask = np.isin(regime_labels[train_indices],
                          [my_regime - 1, my_regime, my_regime + 1])
    local_train_idx = train_indices[regime_mask]
    
    if len(local_train_idx) < 10:
        # Fallback to global k-NN if regime too small
        local_train_idx = train_indices
    
    knn_local = NearestNeighbors(n_neighbors=min(k, len(local_train_idx) - 1))
    knn_local.fit(X_sc[local_train_idx])
    dists, nbr_idx = knn_local.kneighbors(X_sc[i:i+1])
    
    # CI from regime-local neighbors
    nbr_mfe = oracle_signed_mfe[local_train_idx[nbr_idx[0, 1:]]]
    # ... same CI computation as U/V ...

# 3. Compare against global U
results_global = run_knn_ci(X, oracle_signed_mfe, k=50)  # standard U
```

### Variants

```python
# Variant A: Same regime only (strict)
# Variant B: Same regime ± 1 adjacent (relaxed — captures transitions)
# Variant C: Same dp/dt tercile (Analysis E grouping — DOWN/FLAT/UP)
# Variant D: Same volatility quintile (high-vol neighbors for high-vol queries)

for variant in ['strict_regime', 'adjacent_regime', 'dpdt_group', 'vol_quintile']:
    run_experiment(variant)
```

### Output Table

```
Constraint           Dir Acc   CI Width   CI 50%   CI 80%   vs Global U
────────────────────────────────────────────────────────────────────────
Global (U baseline)  ?.?%      ?.?t       ?.?%     ?.?%     —
Same regime          ?.?%      ?.?t       ?.?%     ?.?%     +?.?%
±1 regime            ?.?%      ?.?t       ?.?%     ?.?%     +?.?%
dp/dt tercile        ?.?%      ?.?t       ?.?%     ?.?%     +?.?%
Vol quintile         ?.?%      ?.?t       ?.?%     ?.?%     +?.?%
```

### Gate

```
PROMOTE if: Any variant beats global U by >= 3pp direction accuracy
            AND CI width is tighter (lower median)
            → Use regime-conditional k-NN in production

KILL if:    All variants within 2pp of global
            → Regime conditioning doesn't help, global is fine

NOTE:       If regime-local has fewer neighbors, CI may be WIDER but
            more ACCURATE. Track both metrics.
```

---

## Analysis Z: Fractal Dimension Features (Shi 2018)

### Why

The journal reviewed Shi 2018 in detail (5 fractal dimensions, Random Forest
hit 96% classification on signal types). The paper's analogy maps directly:
their 8 modulation types = our price shape segments, their SNR axis = our
bars-from-entry axis. Never implemented.

### Script: `scripts/research/analysis_z_fractal_dims.py`

### Fractal Dimension Implementations

```python
import numpy as np

def box_dimension(segment: np.ndarray, scales: list = None) -> float:
    """Box-counting dimension. Best overall discriminator (Shi 2018)."""
    if scales is None:
        scales = [2, 4, 8, 16, 32]
    n = len(segment)
    counts = []
    for s in scales:
        if s >= n:
            continue
        n_boxes = 0
        for start in range(0, n, s):
            end = min(start + s, n)
            chunk = segment[start:end]
            n_boxes += 1  # simplification — count non-empty boxes
            # More precise: grid-based counting
        counts.append((np.log(1.0 / s), np.log(max(1, n_boxes))))
    if len(counts) < 2:
        return 1.0
    x, y = zip(*counts)
    slope, _ = np.polyfit(x, y, 1)
    return max(1.0, min(2.0, slope))

def katz_dimension(segment: np.ndarray) -> float:
    """Katz dimension — ratio of curve length to planar extent."""
    n = len(segment)
    if n < 2:
        return 1.0
    dists = np.abs(np.diff(segment))
    L = np.sum(dists)  # total path length
    d = np.max(np.abs(segment - segment[0]))  # max displacement from start
    if d < 1e-10:
        return 1.0
    return np.log10(n - 1) / (np.log10(n - 1) + np.log10(d / L))

def higuchi_dimension(segment: np.ndarray, k_max: int = 8) -> float:
    """Higuchi dimension — complexity from k-subsampled subsequences."""
    n = len(segment)
    L_k = []
    for k in range(1, min(k_max + 1, n // 2)):
        lengths = []
        for m in range(1, k + 1):
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue
            sub = segment[idx]
            L_m = np.sum(np.abs(np.diff(sub))) * (n - 1) / (len(idx) * k)
            lengths.append(L_m)
        if lengths:
            L_k.append((np.log(1.0 / k), np.log(np.mean(lengths) + 1e-10)))
    if len(L_k) < 2:
        return 1.0
    x, y = zip(*L_k)
    slope, _ = np.polyfit(x, y, 1)
    return max(1.0, min(2.0, slope))

def petrosian_dimension(segment: np.ndarray) -> float:
    """Petrosian dimension — fast approximation from zero-crossings."""
    n = len(segment)
    if n < 3:
        return 1.0
    diff = np.diff(segment)
    n_zero_crossings = np.sum(diff[:-1] * diff[1:] < 0)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_zero_crossings)))

def sevcik_dimension(segment: np.ndarray) -> float:
    """Sevcik dimension — normalize to unit square, measure curve length."""
    n = len(segment)
    if n < 2:
        return 1.0
    # Normalize x to [0, 1]
    x_norm = np.linspace(0, 1, n)
    # Normalize y to [0, 1]
    y_range = np.ptp(segment)
    if y_range < 1e-10:
        return 1.0
    y_norm = (segment - segment.min()) / y_range
    # Curve length
    L = np.sum(np.sqrt(np.diff(x_norm)**2 + np.diff(y_norm)**2))
    return 1.0 + np.log(L) / np.log(2 * (n - 1))
```

### Method

```python
# 1. Cut segments (same as Analysis H/I: 16-bar delta segments)
segments = cut_delta_segments(base_df, seg_len=16)

# 2. Compute 5 fractal dimensions per segment
fractal_features = []
for seg in segments:
    delta = seg - seg[0]  # delta from entry
    fractal_features.append([
        box_dimension(delta),
        katz_dimension(delta),
        higuchi_dimension(delta),
        petrosian_dimension(delta),
        sevcik_dimension(delta),
    ])
F = np.array(fractal_features)  # (N, 5)

# 3. Test A: Do fractal dims separate shape types? (Shi 2018 core claim)
# Use Analysis I's shape labels as ground truth
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    F, shape_labels, test_size=0.3, stratify=shape_labels)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)
# Shi 2018 got 96% — what do we get?

# 4. Test B: Do fractal dims improve direction prediction?
# Append to existing 192D → 197D
X_enriched = np.hstack([X_192d, F])  # 197D
# Run Analysis U's k-NN on 192D vs 197D

# 5. Test C: Do fractal dims help Analysis K's GBM?
from sklearn.ensemble import GradientBoostingClassifier
# Train GBM on 192D vs 197D → direction
```

### Output Table

```
Test                          192D Only   192D + 5 Fractal   Delta
──────────────────────────────────────────────────────────────────
Shape classification (RF)     —           ?.?%                (Shi: 96%)
Direction (k-NN, Analysis U)  ?.?%        ?.?%                +?.?%
Direction (GBM, Analysis K)   ?.?%        ?.?%                +?.?%
Feature importance rank:
  box_dimension               —           #??
  katz_dimension              —           #??
  higuchi_dimension           —           #??
  petrosian_dimension         —           #??
  sevcik_dimension            —           #??
```

### Plots

1. **Fractal dim separation by shape type** (5 box plots, one per dim, colored by shape)
2. **RF confusion matrix** (shape classification from 5 fractal dims)
3. **Feature importance: 197D GBM** (are fractal dims in top 20?)
4. **k-NN direction: 192D vs 197D** (accuracy by consensus bin, overlay)

### Gate

```
PROMOTE if: Shape classification > 85% (RF on 5 fractal dims alone)
            AND direction improves by >= 1pp with enriched features
            → Add fractal dims to the 16D feature vector in production

KILL if:    Shape classification < 70%
            OR direction unchanged/worse with enriched features
            → Fractal dims don't add information beyond what 192D already captures

NOTE:       Even if direction doesn't improve, shape classification has
            standalone value for the seed library integration
```

---

## Analysis AA — Seed Shape Direction Signal

**Source spec:** `docs/JULES_WAVEFORM_SEED_INTEGRATION.md` (Parts 1 + 3)

### Thesis

The waveform research (Analysis I/J) built a 20-shape seed library where each
mathematical template (LINEAR_UP, SYMMETRIC_V_DOWN, etc.) maps to a directional
prediction. This analysis tests: **does classifying 16-bar price segments via
Pearson correlation to seed templates actually predict next-bar direction?**

Analysis J got 92% R² on price prediction from physics features. The seed library
provides a simpler, training-free alternative: if the last 16 bars of 15m closes
look like `EXPONENTIAL_UP` with r > 0.75, predict long.

### What It Does

```python
# 1. Load ATLAS 15m data (resample from 15s if needed)
# 2. Build SeedPrimitiveLibrary (20 shapes, N=16)

from training.seed_library import SeedPrimitiveLibrary
lib = SeedPrimitiveLibrary(N=16)

# 3. Sliding window: classify every 16-bar segment
for i in range(16, len(closes_15m)):
    segment = closes_15m[i-16:i]
    shape_name, corr = lib.classify_trajectory(segment)

    if shape_name == 'NOISE':
        continue

    # 4. Predicted direction from shape
    predicted_dir = lib.DIRECTION_MAP[shape_name]  # 'long'/'short'/None
    if predicted_dir is None:
        continue  # volatility shapes have no directional bias

    # 5. Actual direction = sign of next-bar close delta
    actual_delta = closes_15m[i] - closes_15m[i-1]
    actual_dir = 'long' if actual_delta > 0 else 'short'

    # 6. Score
    correct = (predicted_dir == actual_dir)
```

### Variants

| Variant | Description |
|---------|-------------|
| Base (N=16, 15m) | 16 bars of 15-minute closes |
| N=8, 15m | Shorter window, faster adaptation |
| N=16, 5m | Higher resolution (needs 80 bars → more data) |
| N=16, 1h | Macro shapes (slow-TF alignment) |
| Corr ≥ 0.75 | Default threshold |
| Corr ≥ 0.85 | Strict — fewer but higher-quality signals |
| Corr ≥ 0.90 | Ultra-strict — rare signals |

### Additional Measurements

- **Shape distribution:** How often does each of the 20 shapes appear? Is it 80% NOISE?
- **Per-shape accuracy:** Which shapes predict best? (expect directional shapes > reversals)
- **Correlation to existing direction signals:** Do shape signals agree with TBN conviction?
  If always agreeing → redundant. If orthogonal → high integration value.
- **Shape-to-MFE:** For each shape, what's the average next-16-bar MFE? This feeds
  directly into Analysis CC (E[PnL] prediction).
- **Time-of-day:** Do shapes classify better in trending (morning) vs choppy (lunch)?

### Gate

```
PROMOTE if: Directional shapes (8 UP + 8 DOWN) predict next-bar direction
            with accuracy >= 58% AND sample count >= 200 per shape category
            → Wire as Priority 0.5 in direction hierarchy (per JULES spec Part 3)

KILL if:    Accuracy < 52% (worse than coin flip with noise)
            OR < 50 samples per shape (too rare to be useful)

PARTIAL:    If only a SUBSET of shapes work (e.g., EXPONENTIAL_UP at 65%
            but SYMMETRIC_V at 50%), promote only the useful shapes.
            Update DIRECTION_MAP to exclude bad shapes.
```

### Dependency

- Requires `training/seed_library.py` to exist (Part 1 of JULES_WAVEFORM_SEED_INTEGRATION).
  If not yet extracted, the script should embed the class directly (copy from spec).

---

## Analysis BB — Stacked Multi-TF Direction Model

**Source spec:** `docs/JULES_WAVEFORM_SEED_INTEGRATION.md` (Part 4) +
                 `docs/JULES_EXPECTED_PROFIT_PREDICTOR.md` (Phase 1-2)

### Thesis

Each TF worker produces a 16D physics feature vector. Stacking all 11 TFs gives
a 176D fingerprint of the full market state (11 × 16 = 176). A GradientBoosting
model trained on these 176D vectors should outperform any single-TF direction
signal because it learns cross-TF interactions (e.g., "4h trending up + 15m
pulling back = buy the dip").

The waveform research used 192D (12 TFs × 16D) and got strong results. This
analysis adapts it to the live system's 11-TF worker stack (192D or 176D
depending on whether 4h worker is active).

### What It Does

```python
# 1. Load ATLAS for all 11 TFs
# 2. Build stacked matrix: (N_bars, 11, 16) → flatten to (N_bars, 176)
#    Use existing build_stacked_matrices() from tools/research/data.py

X_stacked = build_stacked_matrices(atlas_data)  # (N, 192) or subset to 176

# 3. Label: direction = sign of forward MFE (next 16 bars)
#    Use oracle trade log if available, else compute from ATLAS closes

# 4. Train/test split: first 70% IS, last 30% OOS (walk-forward, no shuffle)
split = int(len(X_stacked) * 0.7)
X_train, X_test = X_stacked[:split], X_stacked[split:]
y_train, y_test = y_dir[:split], y_dir[split:]

# 5. Train GBM
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate OOS
oos_accuracy = clf.score(X_test, y_test)
oos_proba = clf.predict_proba(X_test)[:, 1]

# 7. Calibration: bin predicted P(long) into deciles
#    For each decile, measure actual long fraction
#    Perfect calibration = diagonal line
```

### Variants

| Variant | Description |
|---------|-------------|
| Full 192D (12 TFs) | All ATLAS timeframes including 4h |
| 176D (11 TFs) | Matches live system's current worker stack |
| Delta features | First-difference of features across consecutive bars |
| Confidence-filtered | Only evaluate on predictions where `|P - 0.5| > 0.15` |
| Per-regime | Train separate models per I-MR regime (ties to Analysis Y) |

### Additional Measurements

- **Feature importance:** Which of the 176 features matter most? Expect 4h/1h z-score
  and velocity to dominate (matches Analysis J importance ranking).
- **Confidence vs accuracy:** Is the model well-calibrated? When it says P(long)=0.8,
  does 80% of the time turn out long?
- **IS vs OOS gap:** If IS accuracy = 85% and OOS = 52%, model is memorizing, not learning.
  Target: OOS within 5pp of IS.
- **Comparison to Analysis U k-NN:** Does GBM beat k-NN on the same 192D features?
  (Analysis U already has baseline numbers to compare against.)
- **Save the model:** If OOS accuracy > 58%, pickle the model. This becomes the
  `direction_model.pkl` referenced in the JULES spec (Priority 0.3 in direction hierarchy).

### Gate

```
PROMOTE if: OOS direction accuracy >= 58% (random = 50%)
            AND IS-OOS gap <= 8pp (not overfitting)
            AND calibration slope between 0.7 and 1.3
            → Save direction_model.pkl to checkpoints/
            → Wire as Priority 0.3 in direction hierarchy

KILL if:    OOS accuracy < 53%
            OR IS-OOS gap > 15pp (severe overfit)

PARTIAL:    If confidence-filtered variant works (accuracy >= 62% at |P-0.5|>0.2)
            but unfiltered doesn't, promote WITH the confidence filter as a gate.
```

---

## Analysis CC — E[PnL] Tick Prediction & Calibration

**Source spec:** `docs/JULES_EXPECTED_PROFIT_PREDICTOR.md`

### Thesis

The system currently answers "should I trade?" (gates) and "which direction?"
(direction hierarchy). It does NOT answer "how many ticks will this move?"
If we can predict **signed expected ticks**, three things unlock:

1. **Entry gate:** Skip trades where E[PnL] < 3 ticks (not worth spread + slippage)
2. **Exit sizing:** TP = predicted ticks, SL = predicted × ratio, instead of generic ATR
3. **Progress tracking:** Mid-trade, compare actual vs predicted → exit early if prediction failing

Analysis J achieved 92% R² on price prediction from physics features. This
analysis operationalizes that finding: train a regression model (not classifier)
on the same features, predicting signed forward ticks, and measure calibration.

### What It Does

```python
# 1. Load ATLAS + oracle trade log (for actual MFE/MAE per trade)
#    If no trade log, compute forward MFE from ATLAS closes directly

# 2. Build feature matrix: 192D stacked physics (same as Analysis BB)
X = build_stacked_matrices(atlas_data)

# 3. Target: signed MFE in ticks over next N bars
#    signed_mfe = max favorable excursion × direction_sign
#    e.g., short trade that moves 20 ticks down = -20 (or +20 depending on sign convention)
for i in range(len(X)):
    forward_closes = closes[i:i+16]  # next 16 bars
    forward_max = max(forward_closes) - closes[i]  # max up move
    forward_min = closes[i] - min(forward_closes)  # max down move
    if forward_max > forward_min:
        y_ticks[i] = forward_max / tick_size   # positive = long opportunity
    else:
        y_ticks[i] = -forward_min / tick_size  # negative = short opportunity

# 4. Train regression model
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42)
reg.fit(X_train, y_train)

# 5. OOS prediction
y_pred = reg.predict(X_test)

# 6. Calibration analysis
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 7. Binned calibration: predicted 5-10 ticks → actual avg X ticks
bins = [(-100, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 100)]
for lo, hi in bins:
    mask = (y_pred >= lo) & (y_pred < hi)
    if mask.sum() > 10:
        print(f"  Predicted [{lo},{hi}): actual avg = {y_test[mask].mean():.1f} ticks, N={mask.sum()}")

# 8. Direction accuracy derived from sign
direction_correct = (np.sign(y_pred) == np.sign(y_test)).mean()
```

### Variants

| Variant | Description |
|---------|-------------|
| Forward window 16 bars | Standard (same as Analysis J) |
| Forward window 8 bars | Shorter horizon, tighter predictions |
| Forward window 32 bars | Longer horizon, larger moves |
| Seed shape input | Append shape features from AA to 192D → 193D+ |
| Template-conditional | Separate model per fractal template cluster |

### Additional Measurements

- **Scatter plot:** Predicted vs actual ticks (should cluster around diagonal)
- **Residual distribution:** Are errors symmetric? Or does model systematically
  underpredict large moves? (important for TP sizing)
- **Direction accuracy from regression:** If sign(predicted) matches sign(actual),
  this doubles as a direction model. Compare to Analysis BB's GBM classifier.
- **Minimum-threshold gating:** At predicted > 5 ticks, what's the WR of actual > 0?
  At predicted > 10 ticks? Find the threshold where predicted ticks reliably indicate
  profitable trades.
- **Prediction-based exit simulation:** Replay trades using predicted_ticks as TP.
  Compare PnL to current ATR-based TP. This is the real payoff metric.

### Gate

```
PROMOTE if: OOS R² >= 0.30 (meaningful predictive power)
            AND direction accuracy from sign >= 58%
            AND binned calibration slope between 0.5 and 1.5
            → Save epnl_model.pkl to checkpoints/
            → Wire into exit_engine: TP/SL from prediction (Phase 4 of E[PnL] spec)
            → Wire into execution_engine: E[PnL] gate (Phase 3 of E[PnL] spec)

KILL if:    OOS R² < 0.10 (no better than mean prediction)
            OR direction from sign < 52%

NOTE:       R² of 0.30 sounds low but is strong for tick prediction.
            Analysis J's 92% was on smoothed price curves, not raw ticks.
            Raw tick-level noise will compress R² significantly.
            The binned calibration matters more than raw R².
```

### Dependency

- Benefits from Analysis BB (can reuse the same stacked feature matrix)
- Benefits from Analysis AA (shape features as additional input)
- If promoted, feeds directly into the E[PnL] spec implementation phases 3-6

---

## Analysis DD — Auto-Level Detection & Level-Proximity Features

**Source spec:** `docs/LEVEL_DETECTOR_SPEC.md`

### Thesis

Moises's natural trading method is level-based: pre-mark horizontal support/resistance
zones, wait for price to interact. The Level Detector spec describes an automatic
version: given a daily range (swing high + low), generate Fibonacci levels, detect
sub-levels via swing detection across 4 TFs, cluster them, and score by temporal
recency and cross-TF reinforcement.

This analysis tests the **research question** before building the full pipeline:
**do trades near automatically-detected levels have higher win rates than trades
in "empty space" between levels?** If yes, level proximity becomes a new entry
gate feature. If no, the full 700-line level detection pipeline isn't worth building.

### What It Does

```python
# 1. Simplified level detection (subset of full spec — just enough to test the thesis)

# 1a. Fib levels from daily range
def compute_fib_levels(range_high, range_low):
    span = range_high - range_low
    ratios = [0.0, 0.236, 0.382, 0.500, 0.618, 0.764, 1.0]
    return [range_low + span * r for r in ratios]

# 1b. Swing detection (simplified — just N-bar high/low)
def detect_swings(df, lookback=5):
    highs = df['high'].rolling(lookback*2+1, center=True).max()
    lows = df['low'].rolling(lookback*2+1, center=True).min()
    swing_highs = df['high'][df['high'] == highs].index
    swing_lows = df['low'][df['low'] == lows].index
    return swing_highs, swing_lows

# 1c. For each trading day in ATLAS:
for day in trading_days:
    # Get prior day's range for fib calculation
    prev_day = get_previous_day(day)
    fib_levels = compute_fib_levels(prev_day.high, prev_day.low)

    # Detect swings on 4h, 1h, 15m over trailing window
    swings_4h = detect_swings(atlas['4h'], lookback=6)
    swings_1h = detect_swings(atlas['1h'], lookback=8)
    swings_15m = detect_swings(atlas['15m'], lookback=10)

    # Merge all detected levels
    all_levels = fib_levels + swings_4h + swings_1h + swings_15m

    # 1d. DBSCAN cluster nearby levels (eps = 8 ticks = 2.0 points)
    from sklearn.cluster import DBSCAN
    level_prices = np.array(all_levels).reshape(-1, 1)
    clusters = DBSCAN(eps=2.0, min_samples=2).fit(level_prices)
    cluster_centers = [level_prices[clusters.labels_ == c].mean()
                       for c in set(clusters.labels_) if c != -1]

# 2. For each trade in oracle_trade_log:
for trade in oracle_trades:
    entry_price = trade['entry_price']

    # Distance to nearest detected level (in ticks)
    distances = [abs(entry_price - level) / tick_size for level in cluster_centers]
    min_distance = min(distances) if distances else 999

    trade['distance_to_level'] = min_distance
    trade['near_level'] = min_distance <= 8  # within 8 ticks = "at a level"

# 3. Compare win rates
near_level_trades = [t for t in trades if t['near_level']]
far_from_level = [t for t in trades if not t['near_level']]
wr_near = sum(1 for t in near_level_trades if t['pnl'] > 0) / len(near_level_trades)
wr_far = sum(1 for t in far_from_level if t['pnl'] > 0) / len(far_from_level)
```

### Variants

| Variant | Description |
|---------|-------------|
| Fib only | Just fib levels from daily range (no swing detection) |
| Fib + swings (4 TFs) | Full multi-TF level detection |
| Proximity 4 ticks | Tighter "near level" threshold |
| Proximity 12 ticks | Looser threshold |
| With fib-proximity bonus | Swings near fib lines get 1.5× confidence (per spec) |
| With temporal decay | Recent swings weighted higher than stale ones |

### Additional Measurements

- **Level density:** How many levels per day? If 50+ → too noisy, need stricter clustering.
  If 3-5 → too few, most trades are "far from level."
- **Distance distribution:** Histogram of entry-to-nearest-level distance across all trades.
  If bimodal (many at 0-4 ticks, many at 20+ ticks) → good separation.
  If uniform → levels don't cluster near entries, weak signal.
- **PnL by distance bin:** Not just WR but average PnL per trade at distance [0-4], [4-8],
  [8-16], [16+]. Expect PnL to decrease with distance.
- **Level accuracy:** For each detected level, did price actually bounce/reject within 4 ticks?
  Target > 80% of levels should have at least one bounce (from LEVEL_DETECTOR_SPEC §8).
- **Fib reinforcement:** When a swing-detected level aligns with a fib level (within 4 ticks),
  is the WR boost stronger? This validates the spec's fib-proximity bonus design.
- **Which TF swings matter most?** Daily swings probably better than 15m swings.
  Rank TF contribution to level quality.

### Gate

```
PROMOTE if: WR at near-level trades >= 5pp higher than far-from-level
            AND near-level average PnL > far-from-level average PnL
            AND at least 30% of trades are "near level" (signal isn't too rare)
            → Build full level detection pipeline (all 10 steps from LEVEL_DETECTOR_SPEC)
            → Add distance_to_level as feature in execution_engine gate cascade
            → Generate price_levels.json for live engine

KILL if:    WR difference < 2pp
            OR near-level N < 50 (too few samples)
            OR level accuracy < 50% (detected levels are noise)

PARTIAL:    If only FIB levels work (not swing sub-levels) → simplify pipeline
            to just fib generation. Skip the 400-line swing/cluster/temporal machinery.
```

### Dependency

- Needs ATLAS data for 15m, 1h, 4h, 1D timeframes
- Needs oracle trade log (for trade entry prices and outcomes)
- Independent of V-Z and AA-CC

---

## Analysis EE — Stop Loss Optimization & Dynamic SL

**Source:** `docs/reference/RESEARCH_JOURNAL.txt` — Pipeline Analysis section, Analysis C (proposed but never run)

### Thesis

The March 1 baseline found stop loss was the #1 PnL drain: 153 trades × -$59 avg =
-$9,058 in IS alone. The exit improvements (tiered giveback, 30m flip, etc.) addressed
some of this, but the fundamental question remains: **is the current SL strategy optimal,
and can physics-informed dynamic stops do better?**

The current system uses ATR-based static SL at entry. But the physics features (z-score,
dmi_diff, hurst) carry information about expected adverse excursion. A trade entered at
z=-2 (extended) has different expected MAE than one at z=0 (neutral).

### What It Does

```python
# 1. Load oracle_trade_log.csv (IS + OOS)
# 2. For each trade, extract: entry SL (ticks), actual MAE (ticks), exit_reason, pnl

# 3. Stop loss hit rate: what fraction of trades hit SL?
sl_hits = [t for t in trades if t['exit_reason'] == 'stop_loss']
sl_rate = len(sl_hits) / len(trades)

# 4. MAE distribution analysis
#    For WINNING trades: how deep did they go before recovering?
#    The max MAE of winners sets the minimum viable SL width.
winners = [t for t in trades if t['result'] == 'WIN']
winner_mae_ticks = [t['oracle_mae'] / tick_size for t in winners]
# percentiles: if 95% of winners had MAE < X ticks, then SL = X captures 95%

# 5. Simulate alternative SL widths
for sl_width in [8, 12, 16, 20, 24, 30, 40]:
    sim_results = []
    for trade in trades:
        mae_ticks = trade['oracle_mae'] / tick_size
        if mae_ticks >= sl_width:
            sim_results.append(-sl_width * tick_value)  # would have been stopped
        else:
            sim_results.append(trade['actual_pnl'])  # original outcome
    sim_wr = sum(1 for r in sim_results if r > 0) / len(sim_results)
    sim_pnl = sum(sim_results)

# 6. Physics-conditional SL: SL width varies by entry conditions
#    Hypothesis: wider SL when z is extreme (high reversion probability),
#    tighter SL when z is near zero (low conviction)
for trade in trades:
    if abs(trade['entry_z_score']) > 1.5:
        dynamic_sl = 24  # extended — give more room
    elif abs(trade['entry_z_score']) > 0.8:
        dynamic_sl = 16  # moderate
    else:
        dynamic_sl = 10  # neutral — tight leash

# 7. Trailing-only simulation: what if we removed fixed SL entirely?
#    Use only: trail stop + max_hold + envelope exits
#    Risk: catastrophic single-trade loss. Measure worst-case draw.

# 8. Breakeven acceleration: move SL to breakeven after N ticks profit
for be_threshold in [4, 8, 12, 16]:
    # If trade reaches +be_threshold ticks, move SL to entry
    pass
```

### Variants

| Variant | Description |
|---------|-------------|
| Fixed SL sweep (8-40 ticks) | Brute force optimal width |
| Physics-conditional SL | z-score/dmi-based dynamic width |
| Trailing-only (no fixed SL) | Measure worst-case exposure |
| Breakeven acceleration | SL → entry after N ticks profit |
| Time-decay SL | Tighten SL after M bars if not in profit |
| Depth-conditional | Wider SL for depth 1-2 (slower TF, bigger moves) |

### Additional Measurements

- **Winner MAE percentiles:** P50, P75, P90, P95 of MAE for winning trades.
  The P95 is the "minimum viable SL" that captures 95% of winners.
- **SL efficiency:** For trades that hit SL, what % would have been winners
  if SL was 4 ticks wider? 8 ticks wider? (measures "just barely stopped out")
- **Time-to-SL:** How many bars until SL hit? If most SL hits are in first
  5 bars → entry timing is the problem, not SL width.
- **PnL-optimal SL:** Plot total PnL vs SL width. Find the peak. Current SL
  is optimal only if it's near the peak.

### Gate

```
PROMOTE if: Any variant improves OOS total PnL by >= 10% AND reduces SL hit rate
            → Update exit_engine SL computation with optimal strategy
            → If physics-conditional wins, add z-score SL scaling to make_position()

KILL if:    Current SL is within 5% of optimal (nothing to gain)
            OR all dynamic variants increase worst-case single-trade loss > 2x

NOTE:       This analysis reads existing trade logs. No ATLAS needed.
            Can run in < 1 minute. Highest effort-to-value ratio of all analyses.
```

---

## Analysis FF — Conviction Calibration Audit

**Source:** `docs/reference/RESEARCH_JOURNAL.txt` — Pipeline Analysis section, Analysis D (proposed but never run)

### Thesis

The March 1 baseline found conviction was NON-PREDICTIVE: winners 0.682 vs losers 0.684
(p=0.41). The latest trade analytics (141 trades) shows conviction at p=0.086 — improved
but still not significant. The gate cascade in ExecutionEngine partially replaced conviction
as the primary filter, but conviction still controls WORKER_BYPASS entries and the TBN
belief aggregation weight.

This analysis asks: **is conviction calibrated? When the system says 0.70 conviction,
does it actually win 70% of the time?** And if not, can we recalibrate it?

### What It Does

```python
# 1. Load oracle_trade_log.csv
# 2. Extract: belief_conviction, result (WIN/LOSS)

# 3. Calibration curve: bin conviction into deciles
import numpy as np
conv = np.array([t['belief_conviction'] for t in trades])
won = np.array([t['result'] == 'WIN' for t in trades])

bins = np.arange(0.60, 0.80, 0.02)  # typical range
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (conv >= lo) & (conv < hi)
    if mask.sum() > 5:
        wr = won[mask].mean()
        print(f"  Conv [{lo:.2f}, {hi:.2f}): WR={wr:.1%}, N={mask.sum()}")
        # Perfect calibration: wr ≈ (lo+hi)/2

# 4. Is conviction adding information beyond gates?
# Compare: WR of high-conviction trades vs low-conviction trades
# If WR is flat across conviction bins → conviction is noise

# 5. What SHOULD conviction correlate with?
# Hypothesis: conviction should correlate with:
#   - dmi_diff magnitude (stronger trend = higher conviction)
#   - z-score magnitude (more extended = stronger signal)
#   - depth (fewer TFs aligned = lower conviction)
#   - w5m_c (5m coherence — fast-TF alignment)
from scipy.stats import pearsonr
for feature in ['dmi_diff', 'entry_z_score', 'entry_depth',
                'entry_coherence', 'entry_adx']:
    f_vals = [t[feature] for t in trades]
    r, p = pearsonr(conv, f_vals)
    print(f"  conviction ~ {feature}: r={r:.3f}, p={p:.4f}")

# 6. Recalibration: can we build a better conviction score?
# Use: features that actually predict wins (dmi_diff, z_score, depth)
# Train: logistic regression → P(win | features) = calibrated conviction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
features = ['dmi_diff', 'entry_z_score', 'entry_depth',
            'entry_coherence', 'entry_adx', 'wave_maturity']
X = np.array([[t[f] for f in features] for t in trades])
y = won.astype(int)
lr = LogisticRegression()
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
print(f"  Recalibrated conviction AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
# If AUC > 0.60, the recalibrated version is significantly better than flat 0.68
```

### Variants

| Variant | Description |
|---------|-------------|
| Raw calibration curve | Conviction bins vs actual WR |
| Feature correlation | What does conviction actually correlate with? |
| Recalibrated logistic | Better conviction from winning features |
| Conviction × depth interaction | Does conviction matter more at certain depths? |
| IS vs OOS stability | Is conviction's predictive power stable? |

### Additional Measurements

- **Conviction range:** What's the actual spread? If 0.65-0.70 (narrow) → not enough
  variance to be useful. Need wider spread for meaningful filtration.
- **Bypass rate:** What % of current trades are WORKER_BYPASS (conviction-only entry)
  vs template-matched? This determines how much conviction matters.
- **Time evolution:** Does conviction accuracy degrade over the session/day?
  Early conviction might be better calibrated than late (worker states fresher).
- **Per-template conviction:** Is conviction calibrated differently for different
  fractal templates? Some templates may have systematically overconfident conviction.

### Gate

```
PROMOTE if: Recalibrated conviction AUC >= 0.62
            AND calibration slope between 0.6 and 1.4
            → Replace belief_conviction computation with recalibrated model
            → Wire new conviction into WORKER_BYPASS threshold

KILL if:    AUC < 0.55 (barely better than random)
            AND conviction range < 0.05 (too narrow to be useful)
            → Accept that conviction is a binary filter (above/below threshold)
              not a probability estimate. Simplify code accordingly.

NOTE:       Like EE, reads existing trade logs. No ATLAS needed. Fast to run.
```

---

## Execution Guide

### Running All 11 Independently

```bash
# Each runs in its own terminal / tmux pane / background job
python scripts/research/analysis_v_trajectory_knn.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_w_partial_bar_u.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_x_counter_trend.py  # reads trade logs, no --data needed
python scripts/research/analysis_y_regime_knn.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_z_fractal_dims.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_aa_seed_shape.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_bb_stacked_direction.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_cc_epnl_prediction.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_dd_level_proximity.py --data DATA/ATLAS_1MONTH
python scripts/research/analysis_ee_stop_loss_opt.py   # reads trade logs, no --data needed
python scripts/research/analysis_ff_conviction_audit.py # reads trade logs, no --data needed

# Or all at once (& backgrounds them):
for script in v_trajectory_knn w_partial_bar_u x_counter_trend y_regime_knn z_fractal_dims \
              aa_seed_shape bb_stacked_direction cc_epnl_prediction dd_level_proximity \
              ee_stop_loss_opt ff_conviction_audit; do
    python scripts/research/analysis_${script}.py --data DATA/ATLAS_1MONTH &
done
wait
echo "All analyses complete"
```

### Running at Scale (270 days — final validation)

```bash
# Only run at scale AFTER 1-month results look promising
python scripts/research/analysis_v_trajectory_knn.py --data DATA/ATLAS --analysis-days 270
```

### Checking Results

```bash
# Quick summary of all completed analyses
for dir in reports/research/*/; do
    echo "=== $(basename $dir) ==="
    head -20 "$dir/results.txt" 2>/dev/null || echo "(not complete)"
    echo
done
```

---

## Priority Order

1. **Analysis V** (trajectory k-NN) — highest potential, extends the proven winner
2. **Analysis W** (partial bar U) — gate to production deployment
3. **Analysis EE** (stop loss optimization) — highest effort-to-value ratio, reads existing logs, runs in < 1 min
4. **Analysis FF** (conviction audit) — same: fast, reads existing logs, answers "is conviction useful?"
5. **Analysis BB** (stacked direction model) — operationalizes the 192D fingerprint
6. **Analysis CC** (E[PnL] prediction) — unlocks prediction-based exits (biggest PnL lever)
7. **Analysis AA** (seed shape) — training-free direction signal, quick to validate
8. **Analysis DD** (level proximity) — validates level-based trading thesis with data
9. **Analysis Y** (regime k-NN) — conditional improvement to U
10. **Analysis X** (counter-trend scale) — niche but high-value if validated
11. **Analysis Z** (fractal dims) — exploratory, lowest urgency

V tells you "is the trajectory idea worth anything?"
W tells you "can we actually deploy what we already have?"
EE+FF are free money — run in seconds on existing data, may find easy PnL improvements
BB+CC tell you "can we predict direction AND magnitude?"
DD tells you "does your natural level-based method have measurable statistical edge?"

Everything else is valuable but not blocking.

---

## Source Spec Cross-Reference

These analyses consolidate research from three Jules specs plus the Pipeline Analysis Journal.
Once promoted/killed, the corresponding spec sections are DONE.

| Analysis | Source | Sections Covered |
|----------|--------|------------------|
| AA | `JULES_WAVEFORM_SEED_INTEGRATION.md` | Parts 1 + 3 (library + shape direction) |
| BB | `JULES_WAVEFORM_SEED_INTEGRATION.md` | Part 4 (stacked direction model) |
| BB | `JULES_EXPECTED_PROFIT_PREDICTOR.md` | Phase 1-2 (worker prediction + aggregation) |
| CC | `JULES_EXPECTED_PROFIT_PREDICTOR.md` | Phases 3-5 (entry gate + exit sizing + progress) |
| DD | `LEVEL_DETECTOR_SPEC.md` | Full spec (if PROMOTED → build pipeline; if KILLED → archive) |
| EE | `RESEARCH_JOURNAL.txt` (Pipeline section) | Analysis C (stop loss optimization — proposed, never run) |
| FF | `RESEARCH_JOURNAL.txt` (Pipeline section) | Analysis D (conviction calibration — proposed, never run) |

**Not covered by research (implementation work, do after promotion):**
- `JULES_WAVEFORM_SEED_INTEGRATION.md` Part 2 (4h worker addition) — infra, not research
- `JULES_WAVEFORM_SEED_INTEGRATION.md` Part 5 (live engine integration) — deployment
- `JULES_EXPECTED_PROFIT_PREDICTOR.md` Phase 6 (re-entry decision) — depends on CC promotion
- `LEVEL_DETECTOR_SPEC.md` Steps 6-10 (overlay, exporter, CLI) — build only if DD promotes

**Pipeline Analysis Journal historical findings (keep as reference, already acted on):**
- Analysis A/A2 (March 1 baseline) — template matching was broken, now fixed
- Analysis B (template matching fix) — addressed by recursive K-Means rework
- Hurst integration conflict — still valid WARNING: don't deploy hurst modulation without working direction model

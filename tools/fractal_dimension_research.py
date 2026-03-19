"""
Fractal Dimension Research -- Shi 2018 fractal features for peak classification.

Computes 5 fractal dimension features on peak approach segments and tests
whether they improve peak outcome classification (reversal/plateau/continuation).

Supports two resolutions:
  --resolution 1s  (default): 600-point segments from 1s ATLAS data (10-min window)
  --resolution 1m  (legacy):  10-point segments from 1m ATLAS data

Tests:
  A) 5 fractal dims alone -> 3-class RF (reversal/continuation/plateau)
  B) 54D existing features vs 59D (54D + 5 fractal) -> accuracy comparison
  C) Binary (reversal vs not) -> precision at thresholds
  D) Feature importance ranking (top 20 from 59D RF)
  E) Kruskal-Wallis H-stat per fractal dim across outcome groups

Usage:
    python tools/fractal_dimension_research.py [--data DATA/ATLAS] [--resolution 1s|1m]

Output (1s):
    reports/research/Z_fractal_dims/results_1s.txt
    reports/research/Z_fractal_dims/fractal_separation_1s.png
    reports/research/Z_fractal_dims/fractal_features_1s.csv

Output (1m):
    reports/research/Z_fractal_dims/results_1m.txt
    reports/research/Z_fractal_dims/fractal_separation_1m.png
    reports/research/Z_fractal_dims/fractal_features_1m.csv
"""

import os
import sys
import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from scipy.stats import kruskal

# ── Windows UTF-8 stdout ────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PEAK_CSV = ROOT / 'reports' / 'findings' / 'peak_template_features.csv'
OUT_DIR = ROOT / 'reports' / 'research' / 'Z_fractal_dims'

# ── Feature column prefixes for the 54D existing features ───────────────
FEATURE_PREFIXES = ('geom_', 'peak_', 'delta_', 'slope_')

# ── Fractal dimension names ─────────────────────────────────────────────
FRACTAL_NAMES = ['frac_box', 'frac_katz', 'frac_higuchi', 'frac_petrosian', 'frac_sevcik']

# ── Resolution-specific config ──────────────────────────────────────────
RESOLUTION_CONFIG = {
    '1s': {
        'tf_dir': '1s',
        'lookback_bars': 600,          # 10 minutes * 60 bars/min
        'min_bars': 100,               # Skip peaks with fewer than this many bars
        'box_scales': [2, 4, 8, 16, 32, 64, 128],
        'higuchi_k_max': 16,
        'label': '1s (600-point, 10-min window)',
    },
    '1m': {
        'tf_dir': '1m',
        'lookback_bars': 10,           # 10 bars @ 1m = 10 minutes
        'min_bars': 10,                # Require full 10 bars
        'box_scales': [2, 4, 8],
        'higuchi_k_max': 4,
        'label': '1m (10-point, 10-min window)',
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Fractal dimension functions (Shi 2018)
# ═══════════════════════════════════════════════════════════════════════

def box_dimension(segment, scales=None):
    """Box-counting dimension."""
    if scales is None:
        scales = [2, 4, 8]
    n = len(segment)
    counts = []
    y_range = np.ptp(segment)
    if y_range < 1e-10:
        return 1.0
    for s in scales:
        if s >= n:
            continue
        n_boxes = 0
        for start in range(0, n, s):
            end = min(start + s, n)
            chunk = segment[start:end]
            y_min, y_max = chunk.min(), chunk.max()
            n_boxes += max(1, int(np.ceil((y_max - y_min) / (y_range / s))))
        counts.append((np.log(1.0 / s), np.log(max(1, n_boxes))))
    if len(counts) < 2:
        return 1.0
    x, y = zip(*counts)
    slope, _ = np.polyfit(x, y, 1)
    return max(1.0, min(2.0, slope))


def katz_dimension(segment):
    """Katz dimension - ratio of curve length to planar extent."""
    n = len(segment)
    if n < 2:
        return 1.0
    dists = np.abs(np.diff(segment))
    L = np.sum(dists)
    d = np.max(np.abs(segment - segment[0]))
    if d < 1e-10:
        return 1.0
    return np.log10(n - 1) / (np.log10(n - 1) + np.log10(d / L))


def higuchi_dimension(segment, k_max=4):
    """Higuchi dimension - complexity from k-subsampled subsequences."""
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


def petrosian_dimension(segment):
    """Petrosian dimension - fast approximation from zero-crossings."""
    n = len(segment)
    if n < 3:
        return 1.0
    diff = np.diff(segment)
    n_zero_crossings = np.sum(diff[:-1] * diff[1:] < 0)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_zero_crossings)))


def sevcik_dimension(segment):
    """Sevcik dimension - normalize to unit square, measure curve length."""
    n = len(segment)
    if n < 2:
        return 1.0
    x_norm = np.linspace(0, 1, n)
    y_range = np.ptp(segment)
    if y_range < 1e-10:
        return 1.0
    y_norm = (segment - segment.min()) / y_range
    L = np.sum(np.sqrt(np.diff(x_norm)**2 + np.diff(y_norm)**2))
    return 1.0 + np.log(L) / np.log(2 * (n - 1))


def compute_all_fractal_dims(segment, box_scales, higuchi_k_max):
    """Compute all 5 fractal dimensions for a price segment."""
    return [
        box_dimension(segment, scales=box_scales),
        katz_dimension(segment),
        higuchi_dimension(segment, k_max=higuchi_k_max),
        petrosian_dimension(segment),
        sevcik_dimension(segment),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def _find_columns(df):
    """Identify timestamp and close columns in a parquet DataFrame."""
    ts_col = None
    for candidate in ['timestamp', 'Timestamp', 'time', 'Time', 'date', 'Date']:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    close_col = None
    for candidate in ['Close', 'close', 'CLOSE']:
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        print(f"ERROR: No close column found. Columns: {list(df.columns)}")
        sys.exit(1)

    return ts_col, close_col


def load_atlas_tf(data_root, tf_dir):
    """Load all ATLAS parquets for a given TF, concat, sort by timestamp."""
    atlas_dir = Path(data_root) / tf_dir
    if not atlas_dir.exists():
        print(f"ERROR: {atlas_dir} does not exist")
        sys.exit(1)
    files = sorted(atlas_dir.glob('*.parquet'))
    if not files:
        print(f"ERROR: No parquet files in {atlas_dir}")
        sys.exit(1)
    print(f"Loading {len(files)} {tf_dir} parquet files...")
    dfs = [pd.read_parquet(f) for f in tqdm(files, desc=f'Loading {tf_dir}')]
    print(f"  Concatenating {len(dfs)} DataFrames...")
    df = pd.concat(dfs, ignore_index=True)
    ts_col, close_col = _find_columns(df)
    print(f"  Sorting {len(df):,} bars by {ts_col}...")
    df = df.sort_values(ts_col).reset_index(drop=True)
    print(f"  Done. {len(df):,} total bars.")
    return df, ts_col, close_col


def load_peak_data():
    """Load peak template features CSV."""
    if not PEAK_CSV.exists():
        print(f"ERROR: {PEAK_CSV} does not exist")
        sys.exit(1)
    print(f"Loading peak data from {PEAK_CSV}...")
    df = pd.read_csv(PEAK_CSV)
    print(f"  Loaded {len(df):,} peaks with {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Compute fractal dimensions
# ═══════════════════════════════════════════════════════════════════════

def compute_fractal_features_1m(peak_df, atlas_df, ts_col, close_col, cfg):
    """Extract segments using peak_bar_idx (1m mode) and compute fractal dims."""
    closes = atlas_df[close_col].values
    lookback = cfg['lookback_bars']
    box_scales = cfg['box_scales']
    higuchi_k_max = cfg['higuchi_k_max']
    min_bars = cfg['min_bars']

    peak_bar_indices = peak_df['peak_bar_idx'].values.astype(int)

    fractal_results = np.full((len(peak_df), 5), np.nan)
    valid_mask = np.ones(len(peak_df), dtype=bool)

    print(f"\nComputing fractal dimensions for {len(peak_df):,} peaks (1m, {lookback}-bar segments)...")
    for i in tqdm(range(len(peak_df)), desc='Fractal dims'):
        idx = peak_bar_indices[i]
        start = idx - lookback
        if start < 0 or idx > len(closes):
            valid_mask[i] = False
            continue
        segment = closes[start:idx].astype(np.float64)
        if len(segment) < min_bars:
            valid_mask[i] = False
            continue
        fractal_results[i] = compute_all_fractal_dims(segment, box_scales, higuchi_k_max)

    frac_df = pd.DataFrame(fractal_results, columns=FRACTAL_NAMES)

    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()
    print(f"  Valid segments: {n_valid:,} ({100*n_valid/len(peak_df):.1f}%)")
    print(f"  Invalid (boundary/short): {n_invalid:,}")

    return frac_df, valid_mask


def compute_fractal_features_1s(peak_df, atlas_df, ts_col, close_col, cfg):
    """Extract 600-point 1s segments by timestamp lookup and compute fractal dims."""
    closes = atlas_df[close_col].values
    timestamps = atlas_df[ts_col].values
    lookback = cfg['lookback_bars']  # 600
    box_scales = cfg['box_scales']
    higuchi_k_max = cfg['higuchi_k_max']
    min_bars = cfg['min_bars']  # 100

    peak_timestamps = peak_df['timestamp'].values

    fractal_results = np.full((len(peak_df), 5), np.nan)
    valid_mask = np.ones(len(peak_df), dtype=bool)
    bars_found = np.zeros(len(peak_df), dtype=int)

    print(f"\nComputing fractal dimensions for {len(peak_df):,} peaks "
          f"(1s, up to {lookback}-point segments, min {min_bars})...")
    for i in tqdm(range(len(peak_df)), desc='Fractal dims (1s)'):
        t_peak = peak_timestamps[i]
        # Window: [t_peak - lookback, t_peak - 1] (600 seconds before the peak)
        t_start = t_peak - lookback
        t_end = t_peak - 1

        idx_start = np.searchsorted(timestamps, t_start, side='left')
        idx_end = np.searchsorted(timestamps, t_end, side='right')

        segment = closes[idx_start:idx_end].astype(np.float64)
        bars_found[i] = len(segment)

        if len(segment) < min_bars:
            valid_mask[i] = False
            continue

        fractal_results[i] = compute_all_fractal_dims(segment, box_scales, higuchi_k_max)

    frac_df = pd.DataFrame(fractal_results, columns=FRACTAL_NAMES)

    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()
    median_bars = int(np.median(bars_found[valid_mask])) if n_valid > 0 else 0
    print(f"  Valid segments: {n_valid:,} ({100*n_valid/len(peak_df):.1f}%)")
    print(f"  Invalid (short/<{min_bars} bars): {n_invalid:,}")
    print(f"  Median bars per segment: {median_bars}")

    return frac_df, valid_mask


# ═══════════════════════════════════════════════════════════════════════
# Classification helpers
# ═══════════════════════════════════════════════════════════════════════

def get_existing_feature_cols(df):
    """Get the 54D existing feature columns."""
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def run_rf_timeseries(X, y, n_splits=5, label=''):
    """Train RandomForest with TimeSeriesSplit, return fold accuracies."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []
    all_y_true = []
    all_y_pred = []
    importances = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42 + fold,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        importances = rf.feature_importances_  # last fold

    return {
        'accs': accs,
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'importances': importances,
    }


def run_rf_binary(X, y, n_splits=5):
    """Train RF for binary classification, return precision at thresholds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_y_true = []
    all_y_proba = []
    accs = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42 + fold,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)

    y_true = np.array(all_y_true)
    y_proba = np.array(all_y_proba)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    precision_at_thresh = {}
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        n_pred = y_pred_t.sum()
        if n_pred > 0:
            prec = precision_score(y_true, y_pred_t, zero_division=0)
            precision_at_thresh[t] = (prec, n_pred)
        else:
            precision_at_thresh[t] = (0.0, 0)

    return {
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'precision_at_thresh': precision_at_thresh,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Fractal dimension research for peak classification')
    parser.add_argument('--data', default='DATA/ATLAS', help='ATLAS data root (default: DATA/ATLAS)')
    parser.add_argument('--resolution', default='1s', choices=['1s', '1m'],
                        help='Data resolution: 1s (600-point segments) or 1m (10-point segments)')
    args = parser.parse_args()

    resolution = args.resolution
    cfg = RESOLUTION_CONFIG[resolution]
    data_root = ROOT / args.data
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── File name suffix based on resolution ─────────────────────────────
    suffix = f'_{resolution}'

    # ── Load data ───────────────────────────────────────────────────────
    peak_df = load_peak_data()
    atlas_df, ts_col, close_col = load_atlas_tf(data_root, cfg['tf_dir'])
    print(f"  ATLAS {cfg['tf_dir']}: {len(atlas_df):,} bars, "
          f"columns: {list(atlas_df.columns[:8])}...")

    # ── Step 1: Compute fractal dims ────────────────────────────────────
    if resolution == '1s':
        frac_df, valid_mask = compute_fractal_features_1s(
            peak_df, atlas_df, ts_col, close_col, cfg)
    else:
        frac_df, valid_mask = compute_fractal_features_1m(
            peak_df, atlas_df, ts_col, close_col, cfg)

    # Filter to valid rows
    peak_valid = peak_df[valid_mask].reset_index(drop=True)
    frac_valid = frac_df[valid_mask].reset_index(drop=True)

    # Drop any remaining NaN in fractal features
    frac_nan_mask = ~frac_valid.isna().any(axis=1)
    peak_valid = peak_valid[frac_nan_mask].reset_index(drop=True)
    frac_valid = frac_valid[frac_nan_mask].reset_index(drop=True)

    print(f"\nFinal dataset: {len(peak_valid):,} peaks with valid fractal dims")

    # Outcome labels
    outcomes = peak_valid['outcome'].values
    unique_outcomes = np.unique(outcomes)
    print(f"Outcome distribution:")
    for o in unique_outcomes:
        n = (outcomes == o).sum()
        print(f"  {o}: {n:,} ({100*n/len(outcomes):.1f}%)")

    # ── Existing 54D features ───────────────────────────────────────────
    feat_cols = get_existing_feature_cols(peak_valid)
    print(f"\nExisting feature columns: {len(feat_cols)}")
    X_54 = peak_valid[feat_cols].values.astype(np.float32)
    # Handle NaN/inf in existing features
    X_54 = np.nan_to_num(X_54, nan=0.0, posinf=0.0, neginf=0.0)

    # Fractal features (clean NaN/inf from degenerate segments)
    X_frac = frac_valid.values.astype(np.float32)
    X_frac = np.nan_to_num(X_frac, nan=1.0, posinf=2.0, neginf=1.0)

    # Combined 59D
    X_59 = np.hstack([X_54, X_frac])

    # Labels (encoded)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(outcomes)

    # ════════════════════════════════════════════════════════════════════
    # Open report
    # ════════════════════════════════════════════════════════════════════
    report_lines = []
    def rprint(s=''):
        print(s)
        report_lines.append(s)

    rprint("=" * 72)
    rprint("FRACTAL DIMENSION RESEARCH -- Peak Outcome Classification")
    rprint(f"Resolution: {cfg['label']}")
    rprint(f"Shi 2018: box, katz, higuchi, petrosian, sevcik")
    rprint(f"Box scales: {cfg['box_scales']}")
    rprint(f"Higuchi k_max: {cfg['higuchi_k_max']}")
    rprint("=" * 72)
    rprint(f"\nDataset: {len(peak_valid):,} peaks with valid "
           f"{cfg['lookback_bars']}-point approach segments")
    rprint(f"Existing features: {len(feat_cols)}D")
    rprint(f"Fractal features: {len(FRACTAL_NAMES)}D")
    rprint(f"Combined: {len(feat_cols) + len(FRACTAL_NAMES)}D")
    rprint(f"\nOutcome distribution:")
    for o in unique_outcomes:
        n = (outcomes == o).sum()
        rprint(f"  {o}: {n:,} ({100*n/len(outcomes):.1f}%)")

    # ── Fractal dim summary stats ───────────────────────────────────────
    rprint(f"\n{'─' * 72}")
    rprint("FRACTAL DIMENSION SUMMARY STATISTICS")
    rprint(f"{'─' * 72}")
    for i, name in enumerate(FRACTAL_NAMES):
        vals = X_frac[:, i]
        rprint(f"  {name:20s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
               f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # ════════════════════════════════════════════════════════════════════
    # Test A: 5 fractal dims alone -> 3-class RF
    # ════════════════════════════════════════════════════════════════════
    rprint(f"\n{'═' * 72}")
    rprint("TEST A: 5 Fractal Dims Only -> 3-Class RF (TimeSeriesSplit, 5 folds)")
    rprint(f"{'═' * 72}")

    res_a = run_rf_timeseries(X_frac, y_encoded, label='fractal-only')
    rprint(f"\n  Per-fold accuracy: {[f'{a:.4f}' for a in res_a['accs']]}")
    rprint(f"  Mean accuracy:  {res_a['mean_acc']:.4f} +/- {res_a['std_acc']:.4f}")

    cm_a = confusion_matrix(res_a['y_true'], res_a['y_pred'])
    rprint(f"\n  Confusion matrix (rows=true, cols=pred):")
    rprint(f"  Classes: {list(le.classes_)}")
    for i, row in enumerate(cm_a):
        rprint(f"    {le.classes_[i]:15s}: {row}")

    acc_fractal_only = res_a['mean_acc']

    # ════════════════════════════════════════════════════════════════════
    # Test B: 54D vs 59D -> 3-class RF
    # ════════════════════════════════════════════════════════════════════
    rprint(f"\n{'═' * 72}")
    rprint("TEST B: 54D vs 59D (54D + 5 Fractal) -> 3-Class RF")
    rprint(f"{'═' * 72}")

    rprint("\n  Training 54D baseline...")
    res_54 = run_rf_timeseries(X_54, y_encoded, label='54D')
    rprint(f"  54D accuracy: {res_54['mean_acc']:.4f} +/- {res_54['std_acc']:.4f}")

    rprint("\n  Training 59D (54D + fractal)...")
    res_59 = run_rf_timeseries(X_59, y_encoded, label='59D')
    rprint(f"  59D accuracy: {res_59['mean_acc']:.4f} +/- {res_59['std_acc']:.4f}")

    delta_acc = res_59['mean_acc'] - res_54['mean_acc']
    rprint(f"\n  Delta (59D - 54D): {delta_acc:+.4f} ({delta_acc*100:+.2f} pp)")

    acc_54d = res_54['mean_acc']
    acc_59d = res_59['mean_acc']

    # ════════════════════════════════════════════════════════════════════
    # Test C: Binary (reversal vs not) -> precision at thresholds
    # ════════════════════════════════════════════════════════════════════
    rprint(f"\n{'═' * 72}")
    rprint("TEST C: Binary Classification (reversal vs not)")
    rprint(f"{'═' * 72}")

    y_binary = (outcomes == 'reversal').astype(int)
    n_rev = y_binary.sum()
    rprint(f"\n  Reversal: {n_rev:,} ({100*n_rev/len(y_binary):.1f}%)  "
           f"Not: {len(y_binary) - n_rev:,}")

    rprint("\n  --- 54D baseline ---")
    res_bin_54 = run_rf_binary(X_54, y_binary)
    rprint(f"  Accuracy: {res_bin_54['mean_acc']:.4f} +/- {res_bin_54['std_acc']:.4f}")
    rprint(f"  {'Threshold':>10s}  {'Precision':>10s}  {'N_predicted':>12s}")
    for t, (prec, n) in res_bin_54['precision_at_thresh'].items():
        rprint(f"  {t:10.2f}  {prec:10.4f}  {n:12,}")

    rprint("\n  --- 59D (54D + fractal) ---")
    res_bin_59 = run_rf_binary(X_59, y_binary)
    rprint(f"  Accuracy: {res_bin_59['mean_acc']:.4f} +/- {res_bin_59['std_acc']:.4f}")
    rprint(f"  {'Threshold':>10s}  {'Precision':>10s}  {'N_predicted':>12s}")
    for t, (prec, n) in res_bin_59['precision_at_thresh'].items():
        rprint(f"  {t:10.2f}  {prec:10.4f}  {n:12,}")

    # ════════════════════════════════════════════════════════════════════
    # Test D: Feature importance (top 20 from 59D)
    # ════════════════════════════════════════════════════════════════════
    rprint(f"\n{'═' * 72}")
    rprint("TEST D: Feature Importance -- Top 20 from 59D RF")
    rprint(f"{'═' * 72}")

    all_feat_names = feat_cols + FRACTAL_NAMES
    importances = res_59['importances']
    feat_imp = sorted(zip(all_feat_names, importances), key=lambda x: -x[1])

    rprint(f"\n  {'Rank':>4s}  {'Feature':>30s}  {'Importance':>12s}")
    rprint(f"  {'─'*4}  {'─'*30}  {'─'*12}")
    for rank, (name, imp) in enumerate(feat_imp[:20], 1):
        marker = ' <-- FRACTAL' if name in FRACTAL_NAMES else ''
        rprint(f"  {rank:4d}  {name:>30s}  {imp:12.6f}{marker}")

    # Where do fractal dims rank?
    rprint(f"\n  Fractal dimension rankings:")
    for name in FRACTAL_NAMES:
        rank = next(i for i, (n, _) in enumerate(feat_imp, 1) if n == name)
        imp = next(v for n, v in feat_imp if n == name)
        rprint(f"    {name:20s}: rank {rank:3d} / {len(all_feat_names)}, importance {imp:.6f}")

    # ════════════════════════════════════════════════════════════════════
    # Test E: Kruskal-Wallis H-stat per fractal dim
    # ════════════════════════════════════════════════════════════════════
    rprint(f"\n{'═' * 72}")
    rprint("TEST E: Kruskal-Wallis H-statistic (3 outcome groups)")
    rprint(f"{'═' * 72}")

    rprint(f"\n  Reference benchmarks from existing features:")
    rprint(f"    geom_norm_range:    H = 5802")
    rprint(f"    peak_log_vol_delta: H = 4244")

    rprint(f"\n  {'Feature':>20s}  {'H-stat':>10s}  {'p-value':>12s}  {'Verdict':>10s}")
    rprint(f"  {'─'*20}  {'─'*10}  {'─'*12}  {'─'*10}")

    kw_results = {}
    for i, name in enumerate(FRACTAL_NAMES):
        groups = [X_frac[outcomes == o, i] for o in unique_outcomes]
        # Skip if any group is constant (degenerate)
        if any(np.std(g) < 1e-10 for g in groups if len(g) > 0):
            rprint(f"  {name:>20s}  {'N/A':>10s}  {'N/A':>12s}  {'DEGENERATE':>10s}")
            kw_results[name] = (0.0, 1.0)
            continue
        try:
            h_stat, p_val = kruskal(*groups)
        except ValueError:
            rprint(f"  {name:>20s}  {'N/A':>10s}  {'N/A':>12s}  {'CONSTANT':>10s}")
            kw_results[name] = (0.0, 1.0)
            continue
        verdict = 'STRONG' if h_stat > 1000 else ('MODERATE' if h_stat > 100 else 'WEAK')
        rprint(f"  {name:>20s}  {h_stat:10.1f}  {p_val:12.2e}  {verdict:>10s}")
        kw_results[name] = (h_stat, p_val)

    # Also compute for top existing features for comparison
    rprint(f"\n  Top existing features (for comparison):")
    top_existing = ['geom_norm_range', 'peak_log_vol_delta', 'geom_slope',
                    'geom_curvature', 'geom_efficiency']
    for fname in top_existing:
        if fname in peak_valid.columns:
            vals = peak_valid[fname].values.astype(np.float64)
            vals = np.nan_to_num(vals, nan=0.0)
            groups = [vals[outcomes == o] for o in unique_outcomes]
            try:
                h_stat, p_val = kruskal(*groups)
                rprint(f"  {fname:>20s}  {h_stat:10.1f}  {p_val:12.2e}")
            except ValueError:
                rprint(f"  {fname:>20s}  {'N/A':>10s}  {'N/A':>12s}")

    # ════════════════════════════════════════════════════════════════════
    # Gate decision
    # ════════════════════════════════════════════════════════════════════
    rprint(f"\n{'═' * 72}")
    rprint("GATE DECISION")
    rprint(f"{'═' * 72}")

    promote = acc_fractal_only > 0.85 or delta_acc >= 0.01
    kill = acc_fractal_only < 0.70 and delta_acc < 0.005

    rprint(f"\n  Fractal-only accuracy:  {acc_fractal_only:.4f} (threshold: >0.85 for promote)")
    rprint(f"  59D vs 54D delta:      {delta_acc:+.4f} (threshold: >=1pp for promote)")
    rprint(f"  54D accuracy:          {acc_54d:.4f}")
    rprint(f"  59D accuracy:          {acc_59d:.4f}")

    if promote:
        rprint(f"\n  >>> VERDICT: PROMOTE -- fractal dims add value <<<")
        if acc_fractal_only > 0.85:
            rprint(f"      Reason: fractal-only accuracy {acc_fractal_only:.4f} > 0.85")
        if delta_acc >= 0.01:
            rprint(f"      Reason: enrichment improves accuracy by {delta_acc*100:.2f}pp >= 1pp")
    elif kill:
        rprint(f"\n  >>> VERDICT: KILL -- fractal dims do not help <<<")
        rprint(f"      Reason: fractal-only accuracy {acc_fractal_only:.4f} < 0.70 "
               f"AND delta {delta_acc*100:.2f}pp < 0.5pp")
    else:
        rprint(f"\n  >>> VERDICT: INCONCLUSIVE -- marginal value, needs more investigation <<<")

    # ════════════════════════════════════════════════════════════════════
    # Save outputs
    # ════════════════════════════════════════════════════════════════════

    # 1. Text report
    report_path = OUT_DIR / f'results{suffix}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\nSaved report: {report_path}")

    # 2. Fractal features CSV
    csv_path = OUT_DIR / f'fractal_features{suffix}.csv'
    out_csv = peak_valid[['peak_bar_idx', 'timestamp', 'outcome']].copy()
    for col in FRACTAL_NAMES:
        out_csv[col] = frac_valid[col].values
    out_csv.to_csv(csv_path, index=False)
    print(f"Saved features: {csv_path}")

    # 3. Plot
    plot_path = OUT_DIR / f'fractal_separation{suffix}.png'
    _make_plots(X_frac, outcomes, unique_outcomes,
                acc_fractal_only, acc_54d, acc_59d, plot_path, resolution)
    print(f"Saved plot: {plot_path}")

    print("\nDone.")


def _make_plots(X_frac, outcomes, unique_outcomes, acc_frac, acc_54, acc_59, path, resolution):
    """2x3 grid: 5 box plots + 1 accuracy bar chart."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Fractal Dimension Research -- Peak Outcome Separation ({resolution})',
                 fontsize=14)

    # Color map for outcomes
    colors = {'reversal': '#e74c3c', 'continuation': '#2ecc71', 'plateau': '#3498db'}
    default_colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    # 5 box plots
    for i, name in enumerate(FRACTAL_NAMES):
        row, col = divmod(i, 3)
        ax = axes[row][col]

        data_by_outcome = []
        labels = []
        box_colors = []
        for j, o in enumerate(unique_outcomes):
            mask = outcomes == o
            data_by_outcome.append(X_frac[mask, i])
            labels.append(o)
            box_colors.append(colors.get(o, default_colors[j % len(default_colors)]))

        bp = ax.boxplot(data_by_outcome, labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Dimension value')
        ax.grid(True, alpha=0.3)

    # Accuracy bar chart
    ax = axes[1][2]
    bar_labels = ['5D Fractal\nOnly', '54D\nExisting', '59D\n(54D+Fractal)']
    bar_values = [acc_frac, acc_54, acc_59]
    bar_colors_acc = ['#e67e22', '#3498db', '#2ecc71']
    bars = ax.bar(bar_labels, bar_values, color=bar_colors_acc, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_title('3-Class Accuracy Comparison', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(bar_values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Funnel Decision Tree — learn what t-10m looks like before a tradeable regime starts.

Uses I-MR auto seeds as labeled ground truth (regime entries with known MFE/MAE).
For each seed, extracts 40-bar (10 min) lookback of market state features from 15s data.
Trains a decision tree to classify "setup developing" vs "nothing happening."

In live mode: run the tree every bar on the rolling 40-bar window.
If it fires → "a trade is developing, narrow the funnel."

Usage:
    python tools/funnel_tree_research.py
    python tools/funnel_tree_research.py --seeds tools/plots/standalone/imr_regimes/regime_segments_full_year.csv
    python tools/funnel_tree_research.py --min-mfe 30 --lookback-bars 40
    python tools/funnel_tree_research.py --data DATA/ATLAS_1WEEK

Output: reports/findings/funnel_tree_YYYYMMDD_HHMMSS.txt
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICK_SIZE = 0.25
TICK_VALUE = 0.50


def load_seeds(csv_path: str, min_mfe: float = 30.0) -> pd.DataFrame:
    """Load I-MR regime seeds and filter by minimum MFE."""
    df = pd.read_csv(csv_path)
    print(f"  Raw seeds: {len(df):,}")

    # Filter: profitable regimes only
    df = df[df['direction'] != 'FLAT'].copy()
    df = df[df['mfe_ticks_1s'] >= min_mfe / TICK_VALUE].copy()
    df = df[df['duration_mins'] >= 1.0].copy()
    print(f"  After filter (MFE>=${min_mfe}, dur>=1m, non-FLAT): {len(df):,}")

    return df


def load_15s_data(data_dir: str, months: list = None) -> pd.DataFrame:
    """Load 15s ATLAS data."""
    from tools.research.data import load_atlas_tf
    df = load_atlas_tf(data_dir, '15s', months=months)
    print(f"  15s bars: {len(df):,}")
    return df


def compute_states(df_15s: pd.DataFrame) -> dict:
    """Compute MarketState for all 15s bars."""
    from core.statistical_field_engine import StatisticalFieldEngine
    engine = StatisticalFieldEngine()
    states = engine.batch_compute_states(df_15s, use_cuda=True)
    states_map = {}
    for s in states:
        states_map[s['bar_idx']] = s['state']
    print(f"  States computed: {len(states_map):,}")
    return states_map


def extract_lookback_features(df_15s: pd.DataFrame, states_map: dict,
                                target_bar_idx: int, lookback_bars: int = 40) -> dict:
    """Extract feature vector from the lookback window before a target bar.

    Returns a dict of temporal features computed over the lookback window:
    - z_score trajectory (slope, mean, std, final)
    - ADX trajectory (slope, mean, final)
    - Hurst trajectory (slope, mean, final)
    - DMI balance trajectory (slope)
    - Momentum trajectory (slope, mean)
    - Velocity trajectory (slope)
    - Price trajectory (slope, efficiency, range)
    """
    start_idx = max(0, target_bar_idx - lookback_bars)
    if start_idx == target_bar_idx:
        return None

    # Collect state features over lookback window
    z_scores = []
    adx_vals = []
    hurst_vals = []
    dmi_diffs = []
    momentums = []
    velocities = []
    sigmas = []

    for i in range(start_idx, target_bar_idx):
        state = states_map.get(i)
        if state is None:
            continue
        z_scores.append(getattr(state, 'z_score', 0.0))
        adx_vals.append(getattr(state, 'adx_strength', 0.0))
        hurst_vals.append(getattr(state, 'hurst_exponent', 0.5))
        dmi_diffs.append(getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0))
        momentums.append(getattr(state, 'momentum', 0.0))
        velocities.append(getattr(state, 'velocity', 0.0))
        sigmas.append(getattr(state, 'regression_sigma', 0.0))

    n = len(z_scores)
    if n < 5:
        return None

    # Price trajectory from df_15s
    price_slice = df_15s['close'].iloc[start_idx:target_bar_idx].values.astype(float)
    if len(price_slice) < 5:
        return None

    # Compute temporal features
    x = np.arange(n)
    features = {}

    # Z-score trajectory
    z = np.array(z_scores)
    features['z_mean'] = float(np.mean(z))
    features['z_std'] = float(np.std(z))
    features['z_final'] = float(z[-1])
    features['z_slope'] = float(np.polyfit(x, z, 1)[0]) if n >= 3 else 0.0
    features['z_abs_mean'] = float(np.mean(np.abs(z)))

    # ADX trajectory
    adx = np.array(adx_vals)
    features['adx_mean'] = float(np.mean(adx))
    features['adx_final'] = float(adx[-1])
    features['adx_slope'] = float(np.polyfit(x, adx, 1)[0]) if n >= 3 else 0.0
    features['adx_above_20'] = float(np.mean(adx > 20))

    # Hurst trajectory
    h = np.array(hurst_vals)
    features['hurst_mean'] = float(np.mean(h))
    features['hurst_final'] = float(h[-1])
    features['hurst_slope'] = float(np.polyfit(x, h, 1)[0]) if n >= 3 else 0.0
    features['hurst_above_50'] = float(np.mean(h > 0.50))

    # DMI balance
    dmi = np.array(dmi_diffs)
    features['dmi_mean'] = float(np.mean(dmi))
    features['dmi_slope'] = float(np.polyfit(x, dmi, 1)[0]) if n >= 3 else 0.0
    features['dmi_sign_changes'] = int(np.sum(np.diff(np.sign(dmi)) != 0))
    features['dmi_final'] = float(dmi[-1])

    # Momentum
    mom = np.array(momentums)
    features['mom_mean'] = float(np.mean(mom))
    features['mom_slope'] = float(np.polyfit(x, mom, 1)[0]) if n >= 3 else 0.0
    features['mom_final'] = float(mom[-1])

    # Velocity
    vel = np.array(velocities)
    features['vel_mean'] = float(np.mean(vel))
    features['vel_slope'] = float(np.polyfit(x, vel, 1)[0]) if n >= 3 else 0.0

    # Sigma (volatility)
    sig = np.array(sigmas)
    features['sigma_mean'] = float(np.mean(sig))
    features['sigma_slope'] = float(np.polyfit(x, sig, 1)[0]) if n >= 3 else 0.0

    # Price trajectory
    p = price_slice
    features['price_slope'] = float(np.polyfit(np.arange(len(p)), p, 1)[0])
    features['price_range'] = float((p.max() - p.min()) / TICK_SIZE)
    # Efficiency: how straight-line was the move? (|net| / total path)
    net_move = abs(p[-1] - p[0])
    total_path = np.sum(np.abs(np.diff(p)))
    features['price_efficiency'] = float(net_move / max(total_path, TICK_SIZE))

    # Lookback bar count (might be less than requested if near start of data)
    features['lookback_bars'] = n

    return features


def build_dataset(df_15s: pd.DataFrame, states_map: dict,
                   seeds: pd.DataFrame, lookback_bars: int = 40,
                   neg_ratio: float = 3.0) -> tuple:
    """Build labeled dataset: positive = seed lookbacks, negative = random non-seed bars."""

    timestamps = df_15s['timestamp'].values.astype(float)

    # Build timestamp → bar_idx map
    ts_to_idx = {}
    for i, ts in enumerate(timestamps):
        ts_to_idx[int(ts)] = i

    # Positive samples: lookback before each seed
    pos_features = []
    pos_meta = []

    for _, seed in tqdm(seeds.iterrows(), total=len(seeds), desc='Positive samples',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        ts_start = float(seed['ts_start'])
        # Find closest bar_idx to seed start
        bar_idx = None
        for offset in range(0, 120):  # search within 2 minutes
            check_ts = int(ts_start) - offset
            if check_ts in ts_to_idx:
                bar_idx = ts_to_idx[check_ts]
                break
            check_ts = int(ts_start) + offset
            if check_ts in ts_to_idx:
                bar_idx = ts_to_idx[check_ts]
                break

        if bar_idx is None or bar_idx < lookback_bars:
            continue

        feat = extract_lookback_features(df_15s, states_map, bar_idx, lookback_bars)
        if feat is None:
            continue

        pos_features.append(feat)
        pos_meta.append({
            'ts': ts_start,
            'bar_idx': bar_idx,
            'direction': seed['direction'],
            'mfe_ticks': float(seed['mfe_ticks_1s']),
            'mae_ticks': float(seed['mae_ticks_1s']),
            'duration_mins': float(seed['duration_mins']),
        })

    print(f"  Positive samples: {len(pos_features):,}")

    # Negative samples: random bars that are NOT near any seed
    seed_timestamps = set()
    for _, seed in seeds.iterrows():
        ts = int(seed['ts_start'])
        # Exclude 20-minute window around each seed
        for offset in range(-600, 600):
            seed_timestamps.add(ts + offset)

    neg_features = []
    neg_meta = []
    n_neg_target = int(len(pos_features) * neg_ratio)

    rng = np.random.RandomState(42)
    eligible_bars = [i for i in range(lookback_bars, len(df_15s))
                     if int(timestamps[i]) not in seed_timestamps]

    if len(eligible_bars) < n_neg_target:
        n_neg_target = len(eligible_bars)

    neg_indices = rng.choice(eligible_bars, size=n_neg_target, replace=False)

    for bar_idx in tqdm(neg_indices, desc='Negative samples',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        feat = extract_lookback_features(df_15s, states_map, bar_idx, lookback_bars)
        if feat is None:
            continue
        neg_features.append(feat)
        neg_meta.append({
            'ts': float(timestamps[bar_idx]),
            'bar_idx': bar_idx,
            'direction': 'none',
            'mfe_ticks': 0,
            'mae_ticks': 0,
            'duration_mins': 0,
        })

    print(f"  Negative samples: {len(neg_features):,}")

    # Combine into DataFrame
    all_features = pos_features + neg_features
    all_labels = [1] * len(pos_features) + [0] * len(neg_features)
    all_meta = pos_meta + neg_meta

    df_feat = pd.DataFrame(all_features)
    df_feat['label'] = all_labels
    df_feat['direction'] = [m['direction'] for m in all_meta]
    df_feat['mfe_ticks'] = [m['mfe_ticks'] for m in all_meta]

    return df_feat, all_meta


def train_and_evaluate(df_feat: pd.DataFrame):
    """Train decision tree and evaluate with cross-validation."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report

    feature_cols = [c for c in df_feat.columns
                    if c not in ('label', 'direction', 'mfe_ticks', 'lookback_bars')]
    X = df_feat[feature_cols].values
    y = df_feat['label'].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X):,} (pos={y.sum():,}, neg={(1-y).sum():,})")

    # Cross-validation
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(dt, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(dt, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(dt, X, y, cv=cv, scoring='recall')

    print(f"\n  5-FOLD CROSS-VALIDATION:")
    print(f"    Accuracy:  {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"    F1:        {f1_scores.mean():.3f} +/- {f1_scores.std():.3f}")
    print(f"    Precision: {precision_scores.mean():.3f} +/- {precision_scores.std():.3f}")
    print(f"    Recall:    {recall_scores.mean():.3f} +/- {recall_scores.std():.3f}")

    # Train on full data for feature importance
    dt.fit(X, y)
    importances = dt.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    print(f"\n  TOP 10 FEATURE IMPORTANCES:")
    for name, imp in feat_imp[:10]:
        print(f"    {name:25s}  {imp:.4f}  {'#' * int(imp * 50)}")

    # Full classification report
    y_pred = dt.predict(X)
    print(f"\n  FULL DATASET CLASSIFICATION REPORT:")
    print(classification_report(y, y_pred, target_names=['No setup', 'Setup developing']))

    # Tree rules (first 3 levels)
    from sklearn.tree import export_text
    tree_rules = export_text(dt, feature_names=feature_cols, max_depth=3)
    print(f"\n  DECISION TREE RULES (depth 3):")
    for line in tree_rules.split('\n')[:30]:
        print(f"    {line}")

    # Analyze: do high-MFE seeds have different tree predictions?
    pos_mask = y == 1
    pos_probs = dt.predict_proba(X[pos_mask])[:, 1]
    mfe_vals = df_feat.loc[pos_mask, 'mfe_ticks'].values

    if len(mfe_vals) > 10:
        high_mfe = mfe_vals >= np.percentile(mfe_vals, 75)
        low_mfe = mfe_vals <= np.percentile(mfe_vals, 25)
        print(f"\n  TREE CONFIDENCE vs MFE QUALITY:")
        print(f"    High MFE (P75+): avg tree prob = {pos_probs[high_mfe].mean():.3f}")
        print(f"    Low MFE (P25-):  avg tree prob = {pos_probs[low_mfe].mean():.3f}")
        print(f"    Gap: {pos_probs[high_mfe].mean() - pos_probs[low_mfe].mean():+.3f}")

    return dt, feature_cols, feat_imp


def main():
    parser = argparse.ArgumentParser(description='Funnel Decision Tree Research')
    parser.add_argument('--seeds',
                        default='tools/plots/standalone/imr_regimes/regime_segments_full_year.csv',
                        help='I-MR regime segments CSV')
    parser.add_argument('--data', default='DATA/ATLAS',
                        help='ATLAS data directory for 15s bars')
    parser.add_argument('--month', default=None,
                        help='Restrict to month (e.g. 2025_03)')
    parser.add_argument('--min-mfe', type=float, default=30.0,
                        help='Minimum MFE in dollars to include seed')
    parser.add_argument('--lookback-bars', type=int, default=40,
                        help='Lookback window (40 bars = 10 min at 15s)')
    parser.add_argument('--neg-ratio', type=float, default=3.0,
                        help='Negative:positive sample ratio')
    args = parser.parse_args()

    print("=" * 70)
    print("  FUNNEL DECISION TREE RESEARCH")
    print(f"  Seeds: {args.seeds}")
    print(f"  Data: {args.data}")
    print(f"  Lookback: {args.lookback_bars} bars ({args.lookback_bars * 15 / 60:.0f} min)")
    print("=" * 70)

    # Load seeds
    print("\n[1] Loading seeds...")
    seeds = load_seeds(args.seeds, min_mfe=args.min_mfe)
    if seeds.empty:
        print("  No seeds found")
        return

    # Load 15s data
    print("\n[2] Loading 15s data...")
    months = [args.month] if args.month else None
    df_15s = load_15s_data(args.data, months=months)
    if df_15s.empty:
        return

    # Compute states
    print("\n[3] Computing market states...")
    states_map = compute_states(df_15s)

    # Build dataset
    print("\n[4] Building dataset...")
    df_feat, meta = build_dataset(df_15s, states_map, seeds,
                                    lookback_bars=args.lookback_bars,
                                    neg_ratio=args.neg_ratio)

    if len(df_feat) < 20:
        print("  Not enough samples")
        return

    # Train and evaluate
    print("\n[5] Training decision tree...")
    dt, feature_cols, feat_imp = train_and_evaluate(df_feat)

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'funnel_tree_{ts}.txt')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Funnel Decision Tree Research\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seeds: {args.seeds} ({len(seeds)} after filter)\n")
        f.write(f"Lookback: {args.lookback_bars} bars\n\n")
        f.write(f"Top features:\n")
        for name, imp in feat_imp[:15]:
            f.write(f"  {name:25s}  {imp:.4f}\n")

    # Save model
    import pickle
    model_path = os.path.join(out_dir, f'funnel_tree_model_{ts}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'tree': dt, 'feature_cols': feature_cols}, f)

    print(f"\n  Report saved: {out_path}")
    print(f"  Model saved: {model_path}")


if __name__ == '__main__':
    main()

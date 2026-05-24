"""
Seed Oracle Trainer — Extract 192D features from manual seeds, train entry classifier.

Uses manually marked trade seeds as ground truth labels, extracts multi-TF
(12 x 16 = 192D) feature vectors at each bar, and trains:
  1. Entry classifier: should I enter here? (binary)
  2. Direction classifier: LONG or SHORT? (on entry bars only)
  3. MFE regressor: expected reward (on entry bars only)

Usage:
    python tools/seed_oracle_trainer.py --seeds DATA/regime_seeds/seeds_2025-07-14_20260313_093809.json
    python tools/seed_oracle_trainer.py --seeds DATA/regime_seeds/seeds_2025-07-14_20260313_093809.json --tag Swing
"""

import sys
import json
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import (
    load_atlas_tf, compute_tf_physics, extract_16d,
    TF_HIERARCHY, TF_SECONDS, TF_LABELS, FEATURE_NAMES,
)
from tools.research.screening import flatten_matrices, pad_to_fixed_depth


def load_seeds(seed_path, tag_filter=None):
    """Load seed JSON and optionally filter by tag."""
    with open(seed_path) as f:
        data = json.load(f)

    seeds = data['seeds']
    if tag_filter:
        seeds = [s for s in seeds if s.get('tag', '') == tag_filter]

    print(f"Loaded {len(seeds)} seeds from {seed_path}"
          f"{f' (tag={tag_filter})' if tag_filter else ''}")
    return seeds, data


def detect_month(seeds):
    """Detect ATLAS month from seed timestamps (e.g., '2025_07')."""
    ts = seeds[0]['ts_start']
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{dt.year}_{dt.month:02d}"


def detect_date(seeds):
    """Get trading date from seed timestamps."""
    ts = seeds[0]['ts_start']
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d')


def build_feature_matrix(atlas_dir, month, seeds, base_tf='1m'):
    """Build 192D feature matrix for all bars on the seed day.

    Returns:
        X: (n_bars, 192) feature matrix
        col_names: list of 192 column names
        bar_timestamps: (n_bars,) array of timestamps
        bar_prices: (n_bars,) array of close prices
    """
    print("\n=== Phase 1: Load ATLAS data ===")

    # Load all TFs for the month
    all_tf_states = {}
    for tf in tqdm(TF_HIERARCHY, desc="Loading TFs"):
        df = load_atlas_tf(atlas_dir, tf, months=[month])
        if df.empty:
            print(f"  {tf}: no data")
            continue
        states = compute_tf_physics(tf, df)
        all_tf_states[tf] = states
        print(f"  {tf}: {len(states)} states")

    if base_tf not in all_tf_states:
        raise RuntimeError(f"No data for base TF {base_tf}")

    # Get base TF data
    base_states = all_tf_states[base_tf]
    base_timestamps = sorted(base_states.keys())

    # Filter to the seed day
    seed_date = detect_date(seeds)
    day_start_ts = seeds[0]['ts_start'] - 3600  # 1h before first seed
    day_end_ts = seeds[-1]['ts_end'] + 3600      # 1h after last seed
    day_timestamps = [t for t in base_timestamps if day_start_ts <= t <= day_end_ts]
    print(f"\n  Seed day: {seed_date}")
    print(f"  Day bars: {len(day_timestamps)} ({base_tf})")

    # Build (n, 12, 16) matrix for all day bars
    print("\n=== Phase 2: Build 192D features ===")
    base_secs = TF_SECONDS.get(base_tf, 60)
    n_bars = len(day_timestamps)
    all_mats = np.zeros((n_bars, 12, 16), dtype=np.float64)

    # Pre-extract per-TF arrays
    tf_ts_arrays = {}
    tf_feat_arrays = {}
    for tf in TF_HIERARCHY:
        if tf not in all_tf_states:
            continue
        sorted_ts = sorted(all_tf_states[tf].keys())
        feats = np.array([extract_16d(all_tf_states[tf][t], tf) for t in sorted_ts])
        tf_ts_arrays[tf] = np.array(sorted_ts, dtype=np.int64)
        tf_feat_arrays[tf] = feats

    # Vectorized alignment (same logic as build_stacked_matrices)
    day_ts_arr = np.array(day_timestamps, dtype=np.int64)
    for depth_idx, tf in enumerate(TF_HIERARCHY):
        if tf not in tf_ts_arrays:
            continue
        tf_ts = tf_ts_arrays[tf]
        tf_feats = tf_feat_arrays[tf]
        tf_secs = TF_SECONDS.get(tf, 60)

        raw_idx = np.searchsorted(tf_ts, day_ts_arr, side='right')
        if tf_secs > base_secs:
            raw_idx -= 2  # N-1 for slow TFs
        else:
            raw_idx -= 1  # current completed bar

        valid = raw_idx >= 0
        if not valid.any():
            continue
        clipped_idx = np.clip(raw_idx, 0, len(tf_ts) - 1)
        all_mats[valid, depth_idx, :] = tf_feats[clipped_idx[valid]]

    # Flatten to 192D
    flat, col_names = flatten_matrices(all_mats)
    print(f"  Feature matrix: {flat.shape}")

    # Get prices
    bar_prices = np.array([base_states[t].price for t in day_timestamps])

    return flat, col_names, np.array(day_timestamps), bar_prices


def label_bars(bar_timestamps, seeds, tolerance_secs=60):
    """Label each bar as entry/non-entry based on seed timestamps.

    Args:
        bar_timestamps: (n,) array of bar timestamps
        seeds: list of seed dicts
        tolerance_secs: match window (±) in seconds

    Returns:
        is_entry: (n,) bool array
        direction: (n,) array of 1=LONG, -1=SHORT, 0=none
        tag: (n,) array of 'Swing'/'Scalp'/''
        mfe_dollars: (n,) array of MFE at entry bars, 0 elsewhere
        matched_seeds: list of (bar_idx, seed) tuples
    """
    n = len(bar_timestamps)
    is_entry = np.zeros(n, dtype=bool)
    direction = np.zeros(n, dtype=int)
    tag = np.full(n, '', dtype=object)
    mfe_dollars = np.zeros(n, dtype=float)
    mae_dollars = np.zeros(n, dtype=float)
    matched = []

    for seed in seeds:
        ts = seed['ts_start']
        # Find closest bar
        diffs = np.abs(bar_timestamps - ts)
        idx = np.argmin(diffs)
        if diffs[idx] <= tolerance_secs:
            is_entry[idx] = True
            direction[idx] = 1 if seed['direction'] == 'LONG' else -1
            tag[idx] = seed.get('tag', 'Swing')
            mfe_dollars[idx] = seed.get('mfe_dollars', 0)
            mae_dollars[idx] = seed.get('mae_dollars', 0)
            matched.append((idx, seed))
        else:
            print(f"  WARNING: seed T{seed['trade_id']} ts={ts} not matched "
                  f"(closest bar {diffs[idx]:.0f}s away)")

    print(f"\n  Matched {len(matched)}/{len(seeds)} seeds to bars")
    print(f"  Entry bars: {is_entry.sum()}, Non-entry: {(~is_entry).sum()}")
    return is_entry, direction, tag, mfe_dollars, mae_dollars, matched


def train_classifiers(X, is_entry, direction, tag, mfe_dollars, col_names):
    """Train entry + direction classifiers.

    Returns dict of results.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, precision_recall_curve)

    results = {}

    # ---- Standardize ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- 1. Entry classifier (binary: enter vs skip) ----
    print("\n" + "=" * 60)
    print("ENTRY CLASSIFIER (enter vs skip)")
    print("=" * 60)

    n_pos = is_entry.sum()
    n_neg = (~is_entry).sum()
    print(f"  Positive (entry): {n_pos}")
    print(f"  Negative (skip):  {n_neg}")
    print(f"  Imbalance ratio:  1:{n_neg // max(n_pos, 1)}")

    y_entry = is_entry.astype(int)

    # L1-regularized logistic (sparse, handles 192D with few samples)
    entry_clf = LogisticRegression(
        penalty='l1', solver='saga', C=0.1,
        class_weight='balanced', max_iter=5000, random_state=42
    )

    # Cross-validate (stratified, 5-fold if enough samples)
    n_folds = min(5, n_pos)
    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(entry_clf, X_scaled, y_entry, cv=cv,
                                 scoring='balanced_accuracy')
        print(f"\n  CV balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Per fold: {[f'{s:.3f}' for s in scores]}")
        results['entry_cv_acc'] = float(scores.mean())

    # Fit on full data for feature analysis
    entry_clf.fit(X_scaled, y_entry)
    y_pred = entry_clf.predict(X_scaled)
    y_prob = entry_clf.predict_proba(X_scaled)[:, 1]

    print(f"\n  Full-data classification report:")
    print(classification_report(y_entry, y_pred, target_names=['Skip', 'Enter']))

    # Top features (by |coefficient|)
    coefs = entry_clf.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[::-1][:20]
    nonzero = (np.abs(coefs) > 1e-6).sum()
    print(f"  Non-zero features: {nonzero} / {len(coefs)} (L1 sparsity)")
    print(f"\n  Top 20 entry features:")
    print(f"  {'Rank':>4} {'Feature':<35} {'Coef':>8} {'|Coef|':>8}")
    print(f"  {'----':>4} {'-------':<35} {'----':>8} {'------':>8}")
    for rank, idx in enumerate(top_idx):
        if np.abs(coefs[idx]) < 1e-6:
            break
        print(f"  {rank+1:>4} {col_names[idx]:<35} {coefs[idx]:>8.4f} {abs(coefs[idx]):>8.4f}")

    results['entry_clf'] = entry_clf
    results['entry_top_features'] = [(col_names[i], float(coefs[i])) for i in top_idx[:20]
                                      if abs(coefs[i]) > 1e-6]

    # ---- 2. Direction classifier (LONG vs SHORT, on entry bars only) ----
    print("\n" + "=" * 60)
    print("DIRECTION CLASSIFIER (LONG vs SHORT, entry bars only)")
    print("=" * 60)

    entry_mask = is_entry
    X_entry = X_scaled[entry_mask]
    y_dir = direction[entry_mask]  # 1=LONG, -1=SHORT

    n_long = (y_dir == 1).sum()
    n_short = (y_dir == -1).sum()
    print(f"  LONG:  {n_long}")
    print(f"  SHORT: {n_short}")

    if n_long >= 2 and n_short >= 2:
        dir_clf = LogisticRegression(
            penalty='l1', solver='saga', C=0.5,
            class_weight='balanced', max_iter=5000, random_state=42
        )

        n_folds_dir = min(5, min(n_long, n_short))
        if n_folds_dir >= 2:
            cv_dir = StratifiedKFold(n_splits=n_folds_dir, shuffle=True, random_state=42)
            scores_dir = cross_val_score(dir_clf, X_entry, y_dir, cv=cv_dir,
                                         scoring='balanced_accuracy')
            print(f"\n  CV balanced accuracy: {scores_dir.mean():.3f} ± {scores_dir.std():.3f}")
            results['dir_cv_acc'] = float(scores_dir.mean())

        dir_clf.fit(X_entry, y_dir)
        y_dir_pred = dir_clf.predict(X_entry)
        print(f"\n  Full-data classification report:")
        print(classification_report(y_dir, y_dir_pred, target_names=['SHORT', 'LONG']))

        # Top direction features
        dir_coefs = dir_clf.coef_[0]
        dir_top_idx = np.argsort(np.abs(dir_coefs))[::-1][:15]
        print(f"  Top 15 direction features:")
        print(f"  {'Rank':>4} {'Feature':<35} {'Coef':>8}  {'Meaning'}")
        print(f"  {'----':>4} {'-------':<35} {'----':>8}  {'-------'}")
        for rank, idx in enumerate(dir_top_idx):
            if np.abs(dir_coefs[idx]) < 1e-6:
                break
            meaning = "+LONG" if dir_coefs[idx] > 0 else "+SHORT"
            print(f"  {rank+1:>4} {col_names[idx]:<35} {dir_coefs[idx]:>8.4f}  {meaning}")

        results['dir_clf'] = dir_clf
    else:
        print("  Not enough samples per class for direction classifier")

    # ---- 3. MFE regressor (expected reward at entry bars) ----
    print("\n" + "=" * 60)
    print("MFE REGRESSOR (expected reward, entry bars only)")
    print("=" * 60)

    y_mfe = mfe_dollars[entry_mask]
    print(f"  MFE range: ${y_mfe.min():.1f} - ${y_mfe.max():.1f}")
    print(f"  MFE mean:  ${y_mfe.mean():.1f} ± ${y_mfe.std():.1f}")

    if len(y_mfe) >= 5:
        mfe_reg = Ridge(alpha=10.0)

        from sklearn.model_selection import cross_val_score as cvs_reg
        from sklearn.model_selection import KFold
        n_folds_mfe = min(5, len(y_mfe))
        if n_folds_mfe >= 2:
            cv_mfe = KFold(n_splits=n_folds_mfe, shuffle=True, random_state=42)
            r2_scores = cvs_reg(mfe_reg, X_entry, y_mfe, cv=cv_mfe, scoring='r2')
            print(f"\n  CV R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
            results['mfe_cv_r2'] = float(r2_scores.mean())

        mfe_reg.fit(X_entry, y_mfe)
        y_mfe_pred = mfe_reg.predict(X_entry)
        ss_res = ((y_mfe - y_mfe_pred) ** 2).sum()
        ss_tot = ((y_mfe - y_mfe.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        print(f"  Full-data R²: {r2:.3f}")
        print(f"  Predicted MFE range: ${y_mfe_pred.min():.1f} - ${y_mfe_pred.max():.1f}")

        results['mfe_reg'] = mfe_reg

    results['scaler'] = scaler
    return results


def save_training_data(X, col_names, bar_timestamps, is_entry, direction, tag,
                       mfe_dollars, mae_dollars, bar_prices, output_dir):
    """Save training matrix to CSV for inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(X, columns=col_names)
    df.insert(0, 'timestamp', bar_timestamps)
    df.insert(1, 'price', bar_prices)
    df.insert(2, 'is_entry', is_entry.astype(int))
    df.insert(3, 'direction', direction)
    df.insert(4, 'tag', tag)
    df.insert(5, 'mfe_dollars', mfe_dollars)
    df.insert(6, 'mae_dollars', mae_dollars)

    out_path = output_dir / 'seed_training_matrix.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  Saved training matrix: {out_path} ({len(df)} rows x {df.shape[1]} cols)")

    # Also save entry-only subset
    entry_df = df[df['is_entry'] == 1]
    entry_path = output_dir / 'seed_entries_only.csv'
    entry_df.to_csv(entry_path, index=False)
    print(f"  Saved entries only:    {entry_path} ({len(entry_df)} rows)")

    return out_path


def print_summary(seeds, matched, results):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("SEED ORACLE TRAINING SUMMARY")
    print("=" * 60)

    print(f"\n  Seeds loaded:     {len(seeds)}")
    print(f"  Seeds matched:    {len(matched)}")

    if 'entry_cv_acc' in results:
        print(f"\n  Entry classifier:")
        print(f"    CV balanced acc:  {results['entry_cv_acc']:.3f}")
        n_feat = len(results.get('entry_top_features', []))
        print(f"    Active features:  {n_feat}")

    if 'dir_cv_acc' in results:
        print(f"\n  Direction classifier:")
        print(f"    CV balanced acc:  {results['dir_cv_acc']:.3f}")

    if 'mfe_cv_r2' in results:
        print(f"\n  MFE regressor:")
        print(f"    CV R²:            {results['mfe_cv_r2']:.3f}")

    # Interpretation
    print("\n  --- Interpretation ---")
    entry_acc = results.get('entry_cv_acc', 0)
    if entry_acc > 0.7:
        print("  Entry classifier shows signal — 192D features CAN distinguish")
        print("  entry points from noise. Worth marking more days.")
    elif entry_acc > 0.55:
        print("  Weak signal. Need more seeds (mark more days) or better features.")
    else:
        print("  No signal yet. 47 seeds may not be enough for 192D features.")
        print("  Try: mark 3-5 more days, or reduce to 16D (1m TF only).")


def main():
    parser = argparse.ArgumentParser(description="Seed Oracle Trainer")
    parser.add_argument('--seeds', required=True, help='Path to seed JSON')
    parser.add_argument('--atlas', default='DATA/ATLAS', help='ATLAS root dir')
    parser.add_argument('--tag', default=None, help='Filter seeds by tag (e.g., Swing)')
    parser.add_argument('--base-tf', default='1m', help='Base timeframe (default: 1m)')
    parser.add_argument('--output', default='DATA/oracle_training',
                        help='Output directory for training data')
    args = parser.parse_args()

    # Load seeds
    seeds, seed_data = load_seeds(args.seeds, tag_filter=args.tag)
    if not seeds:
        print("ERROR: No seeds to train on.")
        return

    # Detect month
    month = detect_month(seeds)
    print(f"  Month: {month}")

    # Build feature matrix
    X, col_names, bar_timestamps, bar_prices = build_feature_matrix(
        args.atlas, month, seeds, base_tf=args.base_tf
    )

    # Label bars
    is_entry, direction, tag, mfe_dollars, mae_dollars, matched = label_bars(
        bar_timestamps, seeds
    )

    # Save training data
    save_training_data(X, col_names, bar_timestamps, is_entry, direction, tag,
                       mfe_dollars, mae_dollars, bar_prices, args.output)

    # Train classifiers
    results = train_classifiers(X, is_entry, direction, tag, mfe_dollars, col_names)

    # Summary
    print_summary(seeds, matched, results)


if __name__ == '__main__':
    main()

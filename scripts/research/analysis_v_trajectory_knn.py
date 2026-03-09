#!/usr/bin/env python3
"""
Analysis V: Trajectory k-NN — State Extrapolation
===================================================
Standalone research script. No core/ or live/ modifications.

THESIS: 8 consecutive 192D state snapshots can predict the 9th state.
State transitions are deterministic (Analysis P: 100% at scale).
Direction falls out from the predicted state's z-scores and velocities.

Method:
  1. Build 192D state matrix for all bars
  2. For each bar i, trajectory = states[i-W:i] (W consecutive snapshots)
  3. Target = states[i] (the NEXT state, not direction)
  4. k-NN: find similar trajectories in training set, average their next-states
  5. Predicted state → extract direction from predicted z-scores/velocities
  6. Compare predicted-state-derived direction vs actual direction

Variants:
  - Window sizes: 4, 8, 12, 16
  - Trajectory encodings: flat, delta, summary
  - Direction extraction: from predicted z-score sign, velocity sign, composite

Gate:
  PROMOTE if state prediction R² >= 0.50 AND derived direction accuracy >= 60%
  KILL if R² < 0.20 OR direction < 55%

Usage:
    python scripts/research/analysis_v_trajectory_knn.py
    python scripts/research/analysis_v_trajectory_knn.py --data DATA/ATLAS_1MONTH
    python scripts/research/analysis_v_trajectory_knn.py --data DATA/ATLAS --analysis-days 270
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.research.data import (
    load_atlas_tf, compute_tf_physics, extract_16d,
    build_stacked_matrices, TF_HIERARCHY, TF_SECONDS, FEATURE_NAMES,
)

ANALYSIS_ID = 'V_trajectory_knn'
OUT_DIR = os.path.join('reports', 'research', ANALYSIS_ID)
os.makedirs(OUT_DIR, exist_ok=True)

TICK = 0.25
N_CORES = max(1, multiprocessing.cpu_count() - 1)  # leave 1 core free


# ---------------------------------------------------------------------------
# Build trajectory features
# ---------------------------------------------------------------------------
def build_trajectories(X, window, variant):
    """Build trajectory feature matrix from sequence of 192D state vectors.

    Returns:
        X_traj: (N - window, D) trajectory features
        valid_indices: indices into X for the target bar (= window, window+1, ...)
    """
    n, d = X.shape
    if n <= window:
        return np.empty((0, 0)), np.array([], dtype=int)

    indices = np.arange(window, n)

    if variant == 'flat':
        trajs = np.array([X[i - window: i].flatten() for i in indices])
    elif variant == 'delta':
        trajs = np.array([
            np.diff(X[i - window: i], axis=0).flatten() for i in indices
        ])
    elif variant == 'summary':
        trajs = []
        for i in indices:
            chunk = X[i - window: i]
            mu = chunk.mean(axis=0)
            sd = chunk.std(axis=0)
            slope = (chunk[-1] - chunk[0]) / max(window - 1, 1)
            trajs.append(np.concatenate([mu, sd, slope]))
        trajs = np.array(trajs)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return trajs, indices


# ---------------------------------------------------------------------------
# Worker function for parallel k-NN prediction
# ---------------------------------------------------------------------------
def _predict_chunk(chunk_indices, X_train_sc, Y_train, X_test_sc, k):
    """Predict next-state for a chunk of test indices. Runs in separate process."""
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X_train_sc)

    X_chunk = X_test_sc[chunk_indices]
    dists, nbr_idx = knn.kneighbors(X_chunk)

    # For each test point: predicted state = weighted mean of neighbor targets
    # Weight by inverse distance (closer neighbors matter more)
    predictions = np.zeros((len(chunk_indices), Y_train.shape[1]))
    for i in range(len(chunk_indices)):
        d = dists[i]
        w = 1.0 / (d + 1e-8)
        w /= w.sum()
        predictions[i] = np.average(Y_train[nbr_idx[i]], axis=0, weights=w)

    return chunk_indices, predictions


# ---------------------------------------------------------------------------
# Run state extrapolation experiment
# ---------------------------------------------------------------------------
def run_state_extrapolation(X_traj, Y_target, X_raw_target, label, k=50,
                            signed_mfe=None):
    """Predict the 9th 192D state from trajectory, then derive direction.

    Args:
        X_traj: (N, D) trajectory features
        Y_target: (N, 192) target states to predict
        X_raw_target: (N, 192) actual states (same as Y_target, for direction extraction)
        label: experiment name
        k: number of neighbors
        signed_mfe: (N,) actual signed MFE for direction ground truth

    Returns:
        dict with metrics
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    n = len(X_traj)
    split = int(n * 0.75)
    if split < k + 10 or n - split < 10:
        print(f"    Skipped {label}: insufficient data")
        return None

    X_train, X_test = X_traj[:split], X_traj[split:]
    Y_train, Y_test = Y_target[:split], Y_target[split:]

    # Scale trajectories (inputs)
    scaler_x = StandardScaler().fit(X_train)
    X_train_sc = np.nan_to_num(scaler_x.transform(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    X_test_sc = np.nan_to_num(scaler_x.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)

    # Scale targets — predict in normalized space, then inverse-transform
    scaler_y = StandardScaler().fit(Y_train)
    Y_train_sc = np.nan_to_num(scaler_y.transform(Y_train), nan=0.0, posinf=0.0, neginf=0.0)
    Y_test_sc = np.nan_to_num(scaler_y.transform(Y_test), nan=0.0, posinf=0.0, neginf=0.0)

    k_actual = min(k, len(X_train) // 10, len(X_train) - 1)
    if k_actual < 5:
        return None

    # ---- Parallel k-NN prediction (in scaled target space) ----
    n_test = len(X_test)
    chunk_size = max(50, n_test // N_CORES)
    chunks = [list(range(i, min(i + chunk_size, n_test)))
              for i in range(0, n_test, chunk_size)]

    print(f"    k-NN {label}: {n_test} test points, {len(chunks)} chunks across {N_CORES} cores")

    Y_pred_sc = np.zeros_like(Y_test_sc)

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {
            executor.submit(_predict_chunk, chunk, X_train_sc, Y_train_sc,
                            X_test_sc, k_actual): chunk
            for chunk in chunks
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"    {label}", ascii=True, leave=False):
            idx, preds = future.result()
            Y_pred_sc[idx] = preds

    # Inverse-transform predictions back to original scale
    Y_pred = scaler_y.inverse_transform(Y_pred_sc)

    # ---- State prediction quality ----
    # Per-feature R² (computed individually to avoid scale issues)
    r2_per_feat = np.array([
        r2_score(Y_test[:, j], Y_pred[:, j])
        if np.std(Y_test[:, j]) > 1e-8 else 0.0
        for j in range(Y_test.shape[1])
    ])
    # Clip extreme negatives for averaging (a few degenerate features shouldn't dominate)
    r2_per_feat_clipped = np.clip(r2_per_feat, -1.0, 1.0)
    r2_overall = r2_per_feat_clipped.mean()
    mae_overall = mean_absolute_error(Y_test, Y_pred, multioutput='uniform_average')

    # Per-TF R² (which TFs are most predictable?)
    r2_per_tf = []
    for tf_idx in range(12):
        cols = slice(tf_idx * 16, (tf_idx + 1) * 16)
        if np.std(Y_test[:, cols]) > 1e-8:
            r2_tf = r2_score(Y_test[:, cols], Y_pred[:, cols],
                             multioutput='uniform_average')
        else:
            r2_tf = 0.0
        r2_per_tf.append(r2_tf)

    # ---- Direction from predicted state ----
    # Feature layout per TF: [z_score, log1p_vol, log1p_mom, coherence, tf_scale,
    #                         depth, parent_ctx, adx, hurst, dmi_diff, ...]
    # z_score is feature 0 in each TF's 16D block

    # Extract predicted z-scores for each TF
    pred_z_per_tf = np.array([Y_pred[:, tf_idx * 16] for tf_idx in range(12)]).T  # (N, 12)
    actual_z_per_tf = np.array([Y_test[:, tf_idx * 16] for tf_idx in range(12)]).T

    # Direction methods:
    # Method 1: Sign of mean predicted z across all TFs
    #   z < 0 → LONG (below mean), z > 0 → SHORT (above mean)
    pred_z_mean = pred_z_per_tf.mean(axis=1)
    pred_dir_z = np.where(pred_z_mean < 0, 1, -1)  # 1=LONG, -1=SHORT

    # Method 2: Sign of predicted velocity (feature 1 = log1p_vol, feature 2 = log1p_mom)
    #   Use momentum (feature 2) from the base TF (depth 11 = index 11)
    pred_mom = Y_pred[:, 11 * 16 + 2]  # base TF momentum
    pred_dir_mom = np.where(pred_mom > np.median(pred_mom), 1, -1)

    # Method 3: Composite — z-score sign weighted by TF (slower TFs = more weight)
    tf_weights = np.array([2**i for i in range(12)], dtype=float)
    tf_weights /= tf_weights.sum()
    weighted_z = (pred_z_per_tf * tf_weights).sum(axis=1)
    pred_dir_composite = np.where(weighted_z < 0, 1, -1)

    # Actual direction from signed MFE (if provided) or from actual z
    if signed_mfe is not None:
        y_smfe_test = signed_mfe[split:]
        actual_dir = np.where(y_smfe_test > 0, 1, -1)
    else:
        actual_z_mean = actual_z_per_tf.mean(axis=1)
        actual_dir = np.where(actual_z_mean < 0, 1, -1)

    dir_acc_z = (pred_dir_z == actual_dir).mean()
    dir_acc_mom = (pred_dir_mom == actual_dir).mean()
    dir_acc_composite = (pred_dir_composite == actual_dir).mean()
    best_dir_acc = max(dir_acc_z, dir_acc_mom, dir_acc_composite)
    best_method = ['z_mean', 'momentum', 'composite'][
        [dir_acc_z, dir_acc_mom, dir_acc_composite].index(best_dir_acc)]

    # ---- Per-depth direction accuracy ----
    # For each TF, use just that TF's predicted z to call direction
    depth_dir_acc = []
    for tf_idx in range(12):
        pred_z_tf = Y_pred[:, tf_idx * 16]
        pred_dir_tf = np.where(pred_z_tf < 0, 1, -1)
        acc = (pred_dir_tf == actual_dir).mean()
        depth_dir_acc.append(acc)

    return {
        'label': label,
        'n_train': split,
        'n_test': n_test,
        'dims_traj': X_traj.shape[1],
        'k': k_actual,
        'r2_overall': r2_overall,
        'mae_overall': mae_overall,
        'r2_per_feat': r2_per_feat,
        'r2_per_tf': r2_per_tf,
        'dir_acc_z': dir_acc_z,
        'dir_acc_mom': dir_acc_mom,
        'dir_acc_composite': dir_acc_composite,
        'best_dir_acc': best_dir_acc,
        'best_method': best_method,
        'depth_dir_acc': depth_dir_acc,
        'Y_test': Y_test,
        'Y_pred': Y_pred,
    }


# ---------------------------------------------------------------------------
# Compute signed MFE oracle
# ---------------------------------------------------------------------------
def compute_signed_mfe(base_df, lookahead):
    """Signed MFE: positive = LONG better, negative = SHORT better."""
    closes = base_df['close'].values.astype(float)
    highs = base_df['high'].values.astype(float)
    lows = base_df['low'].values.astype(float)
    n = len(closes)
    signed_mfe = np.full(n, np.nan)

    for i in range(n - lookahead):
        max_up = highs[i + 1: i + 1 + lookahead].max() - closes[i]
        max_down = closes[i] - lows[i + 1: i + 1 + lookahead].min()
        signed_mfe[i] = max_up - max_down

    return signed_mfe


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(all_results, X_all):
    """Generate analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    best = max(valid, key=lambda r: r['r2_overall'])

    # ---- Plot 1: State prediction R² per TF ----
    fig, ax = plt.subplots(figsize=(10, 6))
    tf_labels = [TF_HIERARCHY[i] for i in range(12)]
    r2_vals = best['r2_per_tf']
    colors = ['#228833' if v > 0.5 else '#CCBB44' if v > 0.2 else '#EE6677'
              for v in r2_vals]
    bars = ax.bar(range(12), r2_vals, color=colors, alpha=0.85)
    ax.set_xticks(range(12))
    ax.set_xticklabels(tf_labels, rotation=45, ha='right')
    ax.set_ylabel('R² (state prediction)')
    ax.set_title(f'State Prediction R² per TF ({best["label"]}, overall R²={best["r2_overall"]:.3f})')
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='PROMOTE threshold')
    ax.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='KILL threshold')
    ax.legend()
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot1_r2_per_tf.png'), dpi=150)
    plt.close()

    # ---- Plot 2: Direction accuracy per depth ----
    fig, ax = plt.subplots(figsize=(10, 6))
    depth_accs = [a * 100 for a in best['depth_dir_acc']]
    colors = ['#228833' if a > 55 else '#CCBB44' if a > 52 else '#EE6677'
              for a in depth_accs]
    bars = ax.bar(range(12), depth_accs, color=colors, alpha=0.85)
    ax.set_xticks(range(12))
    ax.set_xticklabels(tf_labels, rotation=45, ha='right')
    ax.set_ylabel('Direction Accuracy %')
    ax.set_title(f'Direction from Predicted State — Per Depth ({best["label"]})')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(55, color='green', linestyle='--', alpha=0.4, label='55% target')
    ax.legend()
    for bar, a in zip(bars, depth_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{a:.1f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot2_direction_per_depth.png'), dpi=150)
    plt.close()

    # ---- Plot 3: Predicted vs actual state (sample features) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Show z-score prediction for 6 TFs
    sample_tfs = [0, 3, 5, 8, 10, 11]  # 1W, 1h, 15m, 2m, 30s, 15s
    for ax, tf_idx in zip(axes.flat, sample_tfs):
        feat_col = tf_idx * 16  # z-score is feature 0
        actual = best['Y_test'][:, feat_col]
        pred = best['Y_pred'][:, feat_col]
        ax.scatter(pred, actual, alpha=0.1, s=5, color='#4477AA')
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        r2 = best['r2_per_tf'][tf_idx]
        ax.set_title(f'{TF_HIERARCHY[tf_idx]} z-score (R²={r2:.2f})', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle(f'State Extrapolation: Predicted vs Actual z-scores ({best["label"]})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot3_state_scatter.png'), dpi=150)
    plt.close()

    # ---- Plot 4: Variant comparison ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = [r['label'] for r in valid]
    r2s = [r['r2_overall'] for r in valid]
    dir_accs = [r['best_dir_acc'] * 100 for r in valid]

    cmap = {'flat': '#EE7733', 'delta': '#33BBEE', 'summary': '#009988'}
    colors = [cmap.get(l.split('_')[0], '#999999') for l in labels]

    ax1.bar(range(len(labels)), r2s, color=colors, alpha=0.85)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('R² (state prediction)')
    ax1.set_title('State Prediction Quality')
    ax1.axhline(0.5, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(0.2, color='red', linestyle='--', alpha=0.5)

    ax2.bar(range(len(labels)), dir_accs, color=colors, alpha=0.85)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Best Direction Accuracy %')
    ax2.set_title('Derived Direction Accuracy')
    ax2.axhline(60, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(55, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot4_variant_comparison.png'), dpi=150)
    plt.close()

    # ---- Plot 5: Per-feature R² heatmap (12 TFs × 16 features) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    r2_matrix = best['r2_per_feat'].reshape(12, 16)
    im = ax.imshow(r2_matrix, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(16))
    ax.set_xticklabels(FEATURE_NAMES, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(12))
    ax.set_yticklabels(TF_HIERARCHY, fontsize=9)
    ax.set_title(f'State Prediction R² per Feature × TF ({best["label"]})')
    plt.colorbar(im, ax=ax, label='R²')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot5_r2_heatmap.png'), dpi=150)
    plt.close()

    print(f"  Plots saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Analysis V: Trajectory State Extrapolation')
    parser.add_argument('--data', default='DATA/ATLAS_1MONTH',
                        help='ATLAS data directory')
    parser.add_argument('--base-tf', default='15m',
                        help='Base timeframe (default: 15m)')
    parser.add_argument('--context-days', type=int, default=21,
                        help='Warmup/context days')
    parser.add_argument('--analysis-days', type=int, default=0,
                        help='Analysis window days (0=all remaining)')
    parser.add_argument('--k', type=int, default=50,
                        help='k neighbors (default: 50)')
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f'Analysis V: Trajectory k-NN — State Extrapolation (v2)')
    print(f'Data: {args.data}, Base TF: {args.base_tf}, k={args.k}')
    print(f'CPU cores: {N_CORES} (of {multiprocessing.cpu_count()})')
    print('=' * 70)

    # ---- 1. Load data & compute physics ----
    print("\n[1/5] Loading ATLAS data and computing physics...")
    all_tf_states = {}
    for tf in tqdm(TF_HIERARCHY, desc="TF physics", ascii=True):
        df = load_atlas_tf(args.data, tf)
        if df.empty:
            continue
        states = compute_tf_physics(tf, df)
        if states:
            all_tf_states[tf] = states

    base_df = load_atlas_tf(args.data, args.base_tf)
    if base_df.empty:
        print(f"ERROR: No data for base TF {args.base_tf}")
        return

    print(f"  Loaded {len(all_tf_states)} TFs, base has {len(base_df)} bars")

    # ---- 2. Build stacked 192D matrices ----
    print("\n[2/5] Building stacked state matrices...")
    matrices, mfes, maes, meta = build_stacked_matrices(
        all_tf_states, args.base_tf, base_df,
        context_days=args.context_days, analysis_days=args.analysis_days,
    )
    if not matrices:
        print("ERROR: No matrices built.")
        return

    X_all = np.array([m.flatten() for m in matrices])  # (N, 192)
    print(f"  State matrix: {X_all.shape}")

    # ---- 3. Compute signed MFE for direction ground truth ----
    print("\n[3/5] Computing signed MFE oracle...")
    from config.oracle_config import ORACLE_LOOKAHEAD_BARS
    lookahead = ORACLE_LOOKAHEAD_BARS.get(args.base_tf, 16)
    raw_signed_mfe = compute_signed_mfe(base_df, lookahead)

    ts_col = base_df['timestamp'].values
    ts_to_idx_df = {int(ts_col[i]): i for i in range(len(ts_col))}

    y_signed_mfe = np.array([
        raw_signed_mfe[ts_to_idx_df[m['ts']]] if m['ts'] in ts_to_idx_df else np.nan
        for m in meta
    ])

    valid = ~np.isnan(y_signed_mfe)
    X_all = X_all[valid]
    y_signed_mfe = y_signed_mfe[valid]
    print(f"  Valid samples: {len(X_all)}")

    # ---- 4. Run experiments ----
    print("\n[4/5] Running state extrapolation experiments...")
    all_results = []

    windows = [4, 8, 12, 16]
    variants = ['flat', 'delta', 'summary']

    for window in windows:
        for variant in variants:
            label = f"{variant}_w{window}"
            print(f"\n  === {label} ===")

            X_traj, traj_indices = build_trajectories(X_all, window, variant)
            if len(X_traj) == 0:
                print(f"    Skipped: not enough data for window={window}")
                continue

            # Target: the NEXT 192D state (what we're predicting)
            Y_target = X_all[traj_indices]
            y_smfe_aligned = y_signed_mfe[traj_indices]

            print(f"    Trajectories: {X_traj.shape}, targets: {Y_target.shape}")

            result = run_state_extrapolation(
                X_traj, Y_target, X_all[traj_indices],
                label, k=args.k, signed_mfe=y_smfe_aligned)
            all_results.append(result)

    # ---- 5. Results ----
    print("\n[5/5] Compiling results...")
    valid_results = [r for r in all_results if r is not None]

    lines = []
    lines.append("=" * 100)
    lines.append("ANALYSIS V: TRAJECTORY k-NN — STATE EXTRAPOLATION (v2)")
    lines.append(f"Data: {args.data}, Base TF: {args.base_tf}, k={args.k}")
    lines.append(f"Total samples: {len(X_all)}, CPU cores: {N_CORES}")
    lines.append("=" * 100)
    lines.append("")
    lines.append("APPROACH: Predict the 9th 192D state from trajectory of 8 states,")
    lines.append("          then derive direction from the predicted state's features.")
    lines.append("")

    # Summary table
    header = (f"{'Variant':<16} {'TrajD':>6} {'N_test':>7} {'R²_state':>9} "
              f"{'MAE':>8} {'DirZ':>7} {'DirMom':>7} {'DirComp':>8} "
              f"{'BestDir':>8} {'Method':<12}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in valid_results:
        lines.append(
            f"{r['label']:<16} {r['dims_traj']:>6} {r['n_test']:>7} "
            f"{r['r2_overall']:>9.4f} {r['mae_overall']:>8.4f} "
            f"{r['dir_acc_z']*100:>6.1f}% {r['dir_acc_mom']*100:>6.1f}% "
            f"{r['dir_acc_composite']*100:>7.1f}% "
            f"{r['best_dir_acc']*100:>7.1f}% {r['best_method']:<12}")

    # Best result detail
    if valid_results:
        best = max(valid_results, key=lambda r: r['r2_overall'])
        lines.append("")
        lines.append(f"--- BEST: {best['label']} ---")
        lines.append("")

        # Per-TF R²
        lines.append("  Per-TF State Prediction R²:")
        lines.append(f"  {'TF':<6} {'R²':>8} {'DirAcc':>8}")
        lines.append(f"  {'-'*6} {'-'*8} {'-'*8}")
        for i, tf in enumerate(TF_HIERARCHY):
            r2 = best['r2_per_tf'][i]
            da = best['depth_dir_acc'][i] * 100
            marker = " <-- best" if da == max(d * 100 for d in best['depth_dir_acc']) else ""
            lines.append(f"  {tf:<6} {r2:>8.4f} {da:>7.1f}%{marker}")

        # Top 10 most predictable features
        lines.append("")
        lines.append("  Top 10 most predictable features (by R²):")
        feat_names = []
        for tf in TF_HIERARCHY:
            for fn in FEATURE_NAMES:
                feat_names.append(f'{tf}_{fn}')
        top_idx = np.argsort(best['r2_per_feat'])[-10:][::-1]
        for idx in top_idx:
            lines.append(f"    {feat_names[idx]:<30} R²={best['r2_per_feat'][idx]:.4f}")

        # Gate evaluation
        lines.append("")
        lines.append("=" * 70)
        lines.append("GATE EVALUATION")
        lines.append("=" * 70)
        lines.append(f"  State prediction R²:  {best['r2_overall']:.4f}  "
                     f"(PROMOTE >= 0.50, KILL < 0.20)")
        lines.append(f"  Best direction acc:   {best['best_dir_acc']*100:.1f}%  "
                     f"(PROMOTE >= 60%, KILL < 55%)")
        lines.append(f"  Best method:          {best['best_method']}")
        lines.append("")

        promote = best['r2_overall'] >= 0.50 and best['best_dir_acc'] >= 0.60
        kill = best['r2_overall'] < 0.20 or best['best_dir_acc'] < 0.55

        if promote:
            verdict = ">>> PROMOTE: State extrapolation works — predicted states carry direction signal"
        elif kill:
            verdict = ">>> KILL: State prediction too weak or direction not derivable"
        else:
            verdict = ">>> DEFER: Partial state prediction — explore richer trajectory encodings"

        lines.append(verdict)

    results_text = "\n".join(lines)
    print("\n" + results_text)

    with open(os.path.join(OUT_DIR, 'results.txt'), 'w') as f:
        f.write(results_text)

    # Plots
    print("\n  Generating plots...")
    make_plots(all_results, X_all)

    # Journal
    _append_journal(ANALYSIS_ID, results_text)

    elapsed = time.perf_counter() - t0
    print(f'\nCompleted in {elapsed:.1f}s')
    print(f'Results: {OUT_DIR}/results.txt')


def _append_journal(analysis_id, text):
    journal = 'docs/reference/RESEARCH_JOURNAL.txt'
    if not os.path.exists(journal):
        print(f"  Journal not found at {journal}, skipping append.")
        return
    with open(journal, 'a', encoding='utf-8') as f:
        f.write(f'\n\n{"=" * 77}\n')
        f.write(f'ANALYSIS {analysis_id.upper()} (auto-generated {time.strftime("%Y-%m-%d")})\n')
        f.write(f'{"=" * 77}\n\n')
        f.write(text)
    print(f"  Appended to {journal}")


if __name__ == '__main__':
    main()

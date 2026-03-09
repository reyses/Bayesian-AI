#!/usr/bin/env python3
"""
Analysis V: Trajectory k-NN Extrapolation
==========================================
Standalone research script. No core/ or live/ modifications.

Hypothesis: 8 sequential 192D state snapshots (trajectory) predict direction
better than a single 192D snapshot (Analysis U baseline).

Variants tested:
  A) Flat: 8×192D = 1536D concatenation
  B) Delta: 7×192D differences between consecutive snapshots = 1344D
  C) Summary: mean/std/slope of each 192 feature across window = 576D
  Window sizes: 4, 8, 12, 16

Gate:
  PROMOTE if V direction accuracy > U by >= 2pp AND CI coverage within 5pp
  KILL if V <= U OR CI degrades > 10pp

Usage:
    python scripts/research/analysis_v_trajectory_knn.py
    python scripts/research/analysis_v_trajectory_knn.py --data DATA/ATLAS_1MONTH
    python scripts/research/analysis_v_trajectory_knn.py --data DATA/ATLAS --analysis-days 270
"""
import argparse
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.research.data import (
    load_atlas_tf, compute_tf_physics, extract_16d,
    build_stacked_matrices, TF_HIERARCHY, TF_SECONDS, FEATURE_NAMES,
)

ANALYSIS_ID = 'V_trajectory_knn'
OUT_DIR = os.path.join('reports', 'research', ANALYSIS_ID)
os.makedirs(OUT_DIR, exist_ok=True)

TICK = 0.25


# ---------------------------------------------------------------------------
# Oracle: signed MFE (positive = up move dominant, negative = down)
# ---------------------------------------------------------------------------
def compute_signed_mfe(base_df, lookahead):
    """For each bar, compute signed MFE = max_up - max_down over lookahead.

    Positive → LONG is better, Negative → SHORT is better.
    This is direction-agnostic (doesn't depend on z-score assignment).
    """
    closes = base_df['close'].values.astype(float)
    highs = base_df['high'].values.astype(float)
    lows = base_df['low'].values.astype(float)
    n = len(closes)
    signed_mfe = np.full(n, np.nan)

    for i in tqdm(range(n - lookahead), desc="Oracle signed MFE",
                  ascii=True, dynamic_ncols=True, mininterval=0.3):
        future_hi = highs[i + 1: i + 1 + lookahead]
        future_lo = lows[i + 1: i + 1 + lookahead]
        max_up = future_hi.max() - closes[i]
        max_down = closes[i] - future_lo.min()
        signed_mfe[i] = max_up - max_down  # positive = LONG better

    return signed_mfe


# ---------------------------------------------------------------------------
# Build trajectory features
# ---------------------------------------------------------------------------
def build_trajectories(X, window, variant):
    """Build trajectory feature matrix from sequence of 192D state vectors.

    Args:
        X: (N, 192) state matrix
        window: number of consecutive states to use
        variant: 'flat', 'delta', or 'summary'

    Returns:
        X_traj: (N - window, D) trajectory features
        valid_indices: indices into X for the target bar (= window, window+1, ...)
    """
    n, d = X.shape
    if n <= window:
        return np.empty((0, 0)), np.array([], dtype=int)

    indices = np.arange(window, n)

    if variant == 'flat':
        # Concatenate window consecutive states: window * 192D
        trajs = np.array([X[i - window: i].flatten() for i in indices])

    elif variant == 'delta':
        # Differences between consecutive states: (window-1) * 192D
        trajs = np.array([
            np.diff(X[i - window: i], axis=0).flatten() for i in indices
        ])

    elif variant == 'summary':
        # Per-feature statistics across window: mean, std, slope = 3 * 192D
        trajs = []
        for i in indices:
            chunk = X[i - window: i]  # (window, 192)
            mu = chunk.mean(axis=0)
            sd = chunk.std(axis=0)
            # Linear slope via polyfit shortcut: (last - first) / (window - 1)
            slope = (chunk[-1] - chunk[0]) / max(window - 1, 1)
            trajs.append(np.concatenate([mu, sd, slope]))
        trajs = np.array(trajs)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return trajs, indices


# ---------------------------------------------------------------------------
# k-NN experiment
# ---------------------------------------------------------------------------
def run_knn_experiment(X_features, y_signed_mfe, label, k=50):
    """Run k-NN direction prediction + CI experiment.

    Args:
        X_features: (N, D) feature matrix (already scaled)
        y_signed_mfe: (N,) signed MFE targets
        label: experiment name for logging
        k: number of neighbors

    Returns:
        dict with metrics
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    n = len(X_features)
    split = int(n * 0.75)
    if split < k + 10 or n - split < 10:
        return None

    X_train, X_test = X_features[:split], X_features[split:]
    y_train, y_test = y_signed_mfe[:split], y_signed_mfe[split:]

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Replace NaN/inf from scaling
    X_train_sc = np.nan_to_num(X_train_sc, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_sc = np.nan_to_num(X_test_sc, nan=0.0, posinf=0.0, neginf=0.0)

    k_actual = min(k, len(X_train) // 10, len(X_train) - 1)
    if k_actual < 5:
        return None

    knn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', n_jobs=-1)
    knn.fit(X_train_sc)

    dists, indices = knn.kneighbors(X_test_sc)

    results = []
    for i in tqdm(range(len(X_test)), desc=f"  k-NN {label}",
                  ascii=True, leave=False, mininterval=0.5):
        nbr_mfe = y_train[indices[i]]

        p10, p25, p50, p75, p90 = np.percentile(nbr_mfe, [10, 25, 50, 75, 90])
        predicted_dir = 'LONG' if p50 > 0 else 'SHORT'
        actual_dir = 'LONG' if y_test[i] > 0 else 'SHORT'

        nbr_long_pct = np.mean(nbr_mfe > 0)
        consensus = max(nbr_long_pct, 1 - nbr_long_pct)

        results.append({
            'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
            'predicted_dir': predicted_dir, 'actual_dir': actual_dir,
            'actual_signed_mfe': y_test[i],
            'consensus': consensus,
            'ci_width': p75 - p25,
            'mean_dist': np.mean(dists[i]),
        })

    rdf = pd.DataFrame(results)

    # Direction accuracy
    dir_correct = (rdf['predicted_dir'] == rdf['actual_dir']).sum()
    dir_acc = dir_correct / len(rdf)
    baseline = max((rdf['actual_dir'] == 'LONG').sum(),
                   (rdf['actual_dir'] == 'SHORT').sum()) / len(rdf)
    lift = dir_acc - baseline

    # CI coverage
    ci50 = ((rdf['p25'] <= rdf['actual_signed_mfe']) &
            (rdf['actual_signed_mfe'] <= rdf['p75'])).mean()
    ci80 = ((rdf['p10'] <= rdf['actual_signed_mfe']) &
            (rdf['actual_signed_mfe'] <= rdf['p90'])).mean()

    # High-consensus accuracy
    hi_cons = rdf[rdf['consensus'] >= 0.90]
    hi_cons_acc = (hi_cons['predicted_dir'] == hi_cons['actual_dir']).mean() \
        if len(hi_cons) > 0 else 0.0

    ci_width_med = rdf['ci_width'].median()

    return {
        'label': label,
        'n_train': split,
        'n_test': len(rdf),
        'dims': X_features.shape[1],
        'k': k_actual,
        'dir_accuracy': dir_acc,
        'baseline': baseline,
        'lift': lift,
        'ci50_coverage': ci50,
        'ci80_coverage': ci80,
        'ci_width_median': ci_width_med,
        'hi_cons_n': len(hi_cons),
        'hi_cons_acc': hi_cons_acc,
        'mean_dist': rdf['mean_dist'].mean(),
        'results_df': rdf,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(all_results, single_point_result):
    """Generate the 4 required plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Find best trajectory result
    traj_results = [r for r in all_results if r and 'Single' not in r['label']]
    if not traj_results:
        print("  No trajectory results to plot.")
        return
    best = max(traj_results, key=lambda r: r['dir_accuracy'])
    sp = single_point_result

    # ---- Plot 1: Direction accuracy by consensus bin (V vs U) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

    for result, color, offset, name in [
        (sp, '#4477AA', -0.15, f'Single-Point (U)'),
        (best, '#EE7733', 0.15, f'Trajectory ({best["label"]})')
    ]:
        if result is None:
            continue
        rdf = result['results_df']
        accs = []
        counts = []
        for j in range(len(bins) - 1):
            mask = (rdf['consensus'] >= bins[j]) & (rdf['consensus'] < bins[j + 1])
            subset = rdf[mask]
            if len(subset) > 0:
                acc = (subset['predicted_dir'] == subset['actual_dir']).mean()
                accs.append(acc * 100)
                counts.append(len(subset))
            else:
                accs.append(0)
                counts.append(0)

        x = np.arange(len(bin_labels))
        bars = ax.bar(x + offset, accs, width=0.28, label=name, color=color, alpha=0.85)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'n={cnt}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Neighbor Consensus')
    ax.set_ylabel('Direction Accuracy %')
    ax.set_title('V vs U: Direction Accuracy by Consensus Bin')
    ax.legend()
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot1_accuracy_by_consensus.png'), dpi=150)
    plt.close()

    # ---- Plot 2: CI calibration curve ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for result, color, name in [
        (sp, '#4477AA', 'Single-Point (U)'),
        (best, '#EE7733', f'Trajectory ({best["label"]})')
    ]:
        if result is None:
            continue
        rdf = result['results_df']
        nominal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        actual_cov = []
        for p in nominal:
            lo_q = (1 - p) / 2 * 100
            hi_q = (1 + p) / 2 * 100
            lo_vals = np.percentile(rdf[['p10', 'p25', 'p50', 'p75', 'p90']].values,
                                     lo_q, axis=1)
            hi_vals = np.percentile(rdf[['p10', 'p25', 'p50', 'p75', 'p90']].values,
                                     hi_q, axis=1)
            cov = ((rdf['actual_signed_mfe'].values >= lo_vals) &
                   (rdf['actual_signed_mfe'].values <= hi_vals)).mean()
            actual_cov.append(cov)
        ax.plot(nominal, actual_cov, 'o-', color=color, label=name)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Nominal CI Coverage')
    ax.set_ylabel('Actual Coverage')
    ax.set_title('CI Calibration: V vs U')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot2_ci_calibration.png'), dpi=150)
    plt.close()

    # ---- Plot 3: P50 vs actual scatter ----
    fig, ax = plt.subplots(figsize=(8, 8))
    if sp:
        ax.scatter(sp['results_df']['p50'], sp['results_df']['actual_signed_mfe'],
                   alpha=0.15, s=8, color='gray', label='Single-Point (U)')
    ax.scatter(best['results_df']['p50'], best['results_df']['actual_signed_mfe'],
               alpha=0.3, s=10, color='#EE7733', label=f'Trajectory ({best["label"]})')
    lims = ax.get_xlim()
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Predicted p50 (signed MFE)')
    ax.set_ylabel('Actual signed MFE')
    ax.set_title('P50 Prediction vs Actual')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot3_p50_vs_actual.png'), dpi=150)
    plt.close()

    # ---- Plot 4: Variant comparison grid ----
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = []
    accs = []
    colors = []
    cmap = {'flat': '#EE7733', 'delta': '#33BBEE', 'summary': '#009988'}
    for r in all_results:
        if r is None or 'Single' in r['label']:
            continue
        labels.append(r['label'])
        accs.append(r['dir_accuracy'] * 100)
        variant = r['label'].split('_')[0] if '_' in r['label'] else 'flat'
        colors.append(cmap.get(variant, '#999999'))

    if labels:
        x = np.arange(len(labels))
        bars = ax.bar(x, accs, color=colors, alpha=0.85)
        if sp:
            ax.axhline(sp['dir_accuracy'] * 100, color='#4477AA',
                        linestyle='--', linewidth=2, label=f'Single-Point baseline ({sp["dir_accuracy"]*100:.1f}%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Direction Accuracy %')
        ax.set_title('All Variants: Direction Accuracy')
        ax.legend()
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=7)
        plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, 'plot4_variant_comparison.png'), dpi=150)
    plt.close()
    print(f"  Plots saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Analysis V: Trajectory k-NN')
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
    print(f'Analysis V: Trajectory k-NN Extrapolation')
    print(f'Data: {args.data}, Base TF: {args.base_tf}, k={args.k}')
    print('=' * 70)

    # ---- 1. Load data & compute physics for all TFs ----
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

    # Flatten (12, 16) -> 192D
    X_all = np.array([m.flatten() for m in matrices])  # (N, 192)
    print(f"  State matrix: {X_all.shape}")

    # ---- 3. Compute signed MFE oracle ----
    print("\n[3/5] Computing signed MFE oracle...")
    from config.oracle_config import ORACLE_LOOKAHEAD_BARS
    lookahead = ORACLE_LOOKAHEAD_BARS.get(args.base_tf, 16)
    raw_signed_mfe = compute_signed_mfe(base_df, lookahead)

    # Map signed_mfe to matrix indices via meta timestamps
    ts_col = base_df['timestamp'].values
    ts_to_idx_df = {int(ts_col[i]): i for i in range(len(ts_col))}

    y_signed_mfe = np.array([
        raw_signed_mfe[ts_to_idx_df[m['ts']]] if m['ts'] in ts_to_idx_df else np.nan
        for m in meta
    ])

    # Drop NaN entries
    valid = ~np.isnan(y_signed_mfe)
    X_all = X_all[valid]
    y_signed_mfe = y_signed_mfe[valid]
    print(f"  Valid samples: {len(X_all)} (signed MFE: "
          f"mean={y_signed_mfe.mean():.2f}, std={y_signed_mfe.std():.2f})")
    n_long = (y_signed_mfe > 0).sum()
    n_short = (y_signed_mfe <= 0).sum()
    print(f"  Direction split: LONG={n_long} ({n_long/len(y_signed_mfe)*100:.1f}%), "
          f"SHORT={n_short} ({n_short/len(y_signed_mfe)*100:.1f}%)")

    # ---- 4. Run experiments ----
    print("\n[4/5] Running k-NN experiments...")
    all_results = []

    # Single-point baseline (Analysis U equivalent)
    print("\n  --- Single-Point Baseline (U) ---")
    sp_result = run_knn_experiment(X_all, y_signed_mfe, "Single-Point (U)", k=args.k)
    all_results.append(sp_result)

    # Trajectory variants
    windows = [4, 8, 12, 16]
    variants = ['flat', 'delta', 'summary']

    for window in windows:
        for variant in variants:
            label = f"{variant}_w{window}"
            print(f"\n  --- {label} ---")

            X_traj, traj_indices = build_trajectories(X_all, window, variant)
            if len(X_traj) == 0:
                print(f"    Skipped: not enough data for window={window}")
                continue

            y_traj = y_signed_mfe[traj_indices]
            print(f"    Features: {X_traj.shape}, samples: {len(y_traj)}")

            result = run_knn_experiment(X_traj, y_traj, label, k=args.k)
            all_results.append(result)

    # ---- 5. Results ----
    print("\n[5/5] Compiling results...")
    valid_results = [r for r in all_results if r is not None]

    # Build comparison table
    lines = []
    lines.append("=" * 90)
    lines.append("ANALYSIS V: TRAJECTORY k-NN EXTRAPOLATION")
    lines.append(f"Data: {args.data}, Base TF: {args.base_tf}, k={args.k}")
    lines.append(f"Total samples: {len(X_all)}, Direction split: "
                 f"LONG={n_long} SHORT={n_short}")
    lines.append("=" * 90)
    lines.append("")

    # Summary table
    header = (f"{'Variant':<22} {'Dims':>6} {'N_test':>7} {'DirAcc':>7} "
              f"{'Baseline':>8} {'Lift':>7} {'CI50':>6} {'CI80':>6} "
              f"{'CIw_med':>8} {'90%+_acc':>8} {'90%+_n':>7}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in valid_results:
        line = (f"{r['label']:<22} {r['dims']:>6} {r['n_test']:>7} "
                f"{r['dir_accuracy']*100:>6.1f}% {r['baseline']*100:>7.1f}% "
                f"{r['lift']*100:>+6.1f}% {r['ci50_coverage']*100:>5.1f}% "
                f"{r['ci80_coverage']*100:>5.1f}% {r['ci_width_median']:>8.2f} "
                f"{r['hi_cons_acc']*100:>7.1f}% {r['hi_cons_n']:>7}")
        lines.append(line)

    lines.append("")

    # Direct comparison: best trajectory vs single-point
    traj_only = [r for r in valid_results if 'Single' not in r['label']]
    if traj_only and sp_result:
        best = max(traj_only, key=lambda r: r['dir_accuracy'])
        lines.append("-" * 70)
        lines.append("HEAD-TO-HEAD: Best Trajectory vs Single-Point (U)")
        lines.append("-" * 70)
        lines.append(f"{'':30} {'Single-Point (U)':>18} {'Trajectory (V)':>18} {'Delta':>10}")
        lines.append(f"{'Direction accuracy':30} {sp_result['dir_accuracy']*100:>17.1f}% "
                     f"{best['dir_accuracy']*100:>17.1f}% "
                     f"{(best['dir_accuracy']-sp_result['dir_accuracy'])*100:>+9.1f}%")
        lines.append(f"{'Lift over baseline':30} {sp_result['lift']*100:>+17.1f}% "
                     f"{best['lift']*100:>+17.1f}% "
                     f"{(best['lift']-sp_result['lift'])*100:>+9.1f}%")
        lines.append(f"{'50% CI coverage':30} {sp_result['ci50_coverage']*100:>17.1f}% "
                     f"{best['ci50_coverage']*100:>17.1f}%")
        lines.append(f"{'80% CI coverage':30} {sp_result['ci80_coverage']*100:>17.1f}% "
                     f"{best['ci80_coverage']*100:>17.1f}%")
        lines.append(f"{'CI width (median)':30} {sp_result['ci_width_median']:>17.2f} "
                     f"{best['ci_width_median']:>17.2f}")
        lines.append(f"{'Consensus 90%+ accuracy':30} {sp_result['hi_cons_acc']*100:>17.1f}% "
                     f"(N={sp_result['hi_cons_n']}) "
                     f"{best['hi_cons_acc']*100:>7.1f}% (N={best['hi_cons_n']})")
        lines.append(f"{'Best variant':30} {'':>18} {best['label']:>18}")

        # Gate evaluation
        lines.append("")
        lines.append("=" * 70)
        lines.append("GATE EVALUATION")
        lines.append("=" * 70)

        delta_acc = (best['dir_accuracy'] - sp_result['dir_accuracy']) * 100
        delta_ci50 = (best['ci50_coverage'] - sp_result['ci50_coverage']) * 100
        delta_ci80 = (best['ci80_coverage'] - sp_result['ci80_coverage']) * 100

        promote = (delta_acc >= 2.0 and
                   abs(delta_ci50) <= 5.0 and
                   best['hi_cons_acc'] >= 0.95 and best['hi_cons_n'] >= 100)
        kill = (delta_acc <= 0.0 or
                delta_ci50 < -10.0 or delta_ci80 < -10.0)
        defer = (delta_acc > 0 and not promote and
                 best['hi_cons_n'] < 100)

        lines.append(f"  Direction accuracy delta:  {delta_acc:+.1f}pp  "
                     f"(need >= +2.0pp for PROMOTE)")
        lines.append(f"  CI50 coverage delta:       {delta_ci50:+.1f}pp  "
                     f"(within ±5pp = OK)")
        lines.append(f"  CI80 coverage delta:       {delta_ci80:+.1f}pp  "
                     f"(degrade > -10pp = KILL)")
        lines.append(f"  90%+ consensus accuracy:   {best['hi_cons_acc']*100:.1f}% "
                     f"with N={best['hi_cons_n']}  (need >= 95% with N>=100)")
        lines.append("")

        if promote:
            verdict = ">>> PROMOTE: Trajectory k-NN adds significant value over single-point"
        elif kill:
            verdict = ">>> KILL: Trajectory k-NN does not improve over single-point"
        elif defer:
            verdict = ">>> DEFER: Promising but insufficient high-consensus samples"
        else:
            verdict = ">>> INCONCLUSIVE: Review manually"
        lines.append(verdict)

    results_text = "\n".join(lines)
    print("\n" + results_text)

    # Write results
    with open(os.path.join(OUT_DIR, 'results.txt'), 'w') as f:
        f.write(results_text)

    # Save detailed CSV
    for r in valid_results:
        if r and 'results_df' in r:
            r['results_df'].to_csv(
                os.path.join(OUT_DIR, f'detail_{r["label"]}.csv'), index=False)

    # Plots
    print("\n  Generating plots...")
    make_plots(all_results, sp_result)

    # Append to journal
    _append_journal(ANALYSIS_ID, results_text)

    elapsed = time.perf_counter() - t0
    print(f'\nCompleted in {elapsed:.1f}s')
    print(f'Results: {OUT_DIR}/results.txt')


def _append_journal(analysis_id, text):
    """Append results to the research journal."""
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

#!/usr/bin/env python3
"""
Analysis CC — E[PnL] Tick Prediction & Calibration
====================================================
Standalone research script. No core/ or live/ modifications.

Trains GBM regression on 192D stacked physics features to predict signed
forward ticks. Tests whether tick predictions are calibrated enough to use
for TP/SL sizing and entry gating.

Variants:
  - Forward windows: 8, 16, 32 bars
  - Models: GBM, Ridge (linear baseline)

Gate:
  PROMOTE if OOS R² >= 0.30 AND direction accuracy >= 58% AND calibration slope 0.5-1.5
  KILL if OOS R² < 0.10 OR direction accuracy < 52%

Usage:
    python scripts/research/analysis_cc_epnl_prediction.py
    python scripts/research/analysis_cc_epnl_prediction.py --data DATA/ATLAS_1MONTH
    python scripts/research/analysis_cc_epnl_prediction.py --data DATA/ATLAS --analysis-days 270
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.research.data import (
    load_atlas_tf, compute_tf_physics, extract_16d,
    build_stacked_matrices, TF_HIERARCHY, TF_SECONDS,
)

ANALYSIS_ID = 'CC_epnl_prediction'
OUT_DIR = os.path.join('reports', 'research', ANALYSIS_ID)
os.makedirs(OUT_DIR, exist_ok=True)

TICK = 0.25


# ---------------------------------------------------------------------------
# Compute signed forward ticks target
# ---------------------------------------------------------------------------
def compute_signed_forward_ticks(base_df, lookahead):
    """For each bar, compute signed forward ticks.

    Positive = LONG opportunity (max up > max down).
    Negative = SHORT opportunity (max down > max up).
    Magnitude = dominant side MFE in ticks.
    """
    closes = base_df['close'].values.astype(float)
    highs = base_df['high'].values.astype(float)
    lows = base_df['low'].values.astype(float)
    n = len(closes)
    y = np.full(n, np.nan)

    for i in range(n - lookahead):
        future_hi = highs[i + 1: i + 1 + lookahead]
        future_lo = lows[i + 1: i + 1 + lookahead]
        max_up = (future_hi.max() - closes[i]) / TICK
        max_down = (closes[i] - future_lo.min()) / TICK

        if max_up > max_down:
            y[i] = max_up    # positive = LONG
        else:
            y[i] = -max_down  # negative = SHORT

    return y


# ---------------------------------------------------------------------------
# Run a single regression experiment
# ---------------------------------------------------------------------------
def run_regression_experiment(X, y, label, model_type='gbm'):
    """Train regression model, evaluate OOS.

    Returns dict with metrics or None on failure.
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    split = int(n * 0.75)
    if split < 100 or n - split < 30:
        print(f"    Skipped {label}: insufficient data (n={n}, split={split})")
        return None

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler().fit(X_train)
    X_train_sc = np.nan_to_num(scaler.transform(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    X_test_sc = np.nan_to_num(scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)

    if model_type == 'gbm':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    print(f"    Training {model_type} for {label}...")
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    # Core metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    direction_acc = (np.sign(y_pred) == np.sign(y_test)).mean()

    # Baseline: always predict mean
    y_mean = np.full_like(y_test, y_train.mean())
    r2_baseline = r2_score(y_test, y_mean)

    # Binned calibration
    bins = [(-200, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 200)]
    bin_data = []
    for lo, hi in bins:
        mask = (y_pred >= lo) & (y_pred < hi)
        n_bin = mask.sum()
        if n_bin >= 5:
            actual_mean = y_test[mask].mean()
            pred_mean = y_pred[mask].mean()
            bin_data.append({
                'bin': f'[{lo},{hi})',
                'n': n_bin,
                'pred_mean': pred_mean,
                'actual_mean': actual_mean,
            })

    # Calibration slope: regress actual_mean on pred_mean across bins
    cal_slope = np.nan
    if len(bin_data) >= 3:
        pred_means = [b['pred_mean'] for b in bin_data]
        actual_means = [b['actual_mean'] for b in bin_data]
        if np.std(pred_means) > 1e-6:
            cal_slope = np.polyfit(pred_means, actual_means, 1)[0]

    # Threshold gating: at predicted > X ticks, what % are actually profitable?
    thresholds = [3, 5, 10, 15, 20]
    threshold_data = []
    for thresh in thresholds:
        # Long trades: predicted > thresh
        long_mask = y_pred > thresh
        if long_mask.sum() >= 5:
            wr = (y_test[long_mask] > 0).mean()
            avg = y_test[long_mask].mean()
            threshold_data.append({
                'threshold': f'>{thresh} (LONG)',
                'n': long_mask.sum(),
                'win_rate': wr,
                'avg_actual': avg,
            })
        # Short trades: predicted < -thresh
        short_mask = y_pred < -thresh
        if short_mask.sum() >= 5:
            wr = (y_test[short_mask] < 0).mean()
            avg = y_test[short_mask].mean()
            threshold_data.append({
                'threshold': f'<-{thresh} (SHORT)',
                'n': short_mask.sum(),
                'win_rate': wr,
                'avg_actual': avg,
            })

    # Feature importance (GBM only)
    feat_imp = None
    if model_type == 'gbm' and hasattr(model, 'feature_importances_'):
        feat_imp = model.feature_importances_

    return {
        'label': label,
        'model_type': model_type,
        'n_train': split,
        'n_test': n - split,
        'r2': r2,
        'r2_baseline': r2_baseline,
        'mae': mae,
        'direction_acc': direction_acc,
        'cal_slope': cal_slope,
        'y_test': y_test,
        'y_pred': y_pred,
        'bin_data': bin_data,
        'threshold_data': threshold_data,
        'feat_imp': feat_imp,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(all_results):
    """Generate analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    # Find the best GBM result
    gbm_results = [r for r in valid if r['model_type'] == 'gbm']
    best = max(gbm_results, key=lambda r: r['r2']) if gbm_results else valid[0]

    # ---- Plot 1: Predicted vs Actual scatter ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(best['y_pred'], best['y_test'], alpha=0.2, s=8, color='#4477AA')
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Predicted signed ticks')
    ax.set_ylabel('Actual signed ticks')
    ax.set_title(f'{best["label"]}: Predicted vs Actual (R²={best["r2"]:.3f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot1_pred_vs_actual.png'), dpi=150)
    plt.close()

    # ---- Plot 2: Residual distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = best['y_test'] - best['y_pred']
    ax.hist(residuals, bins=60, color='#4477AA', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Residual (actual - predicted) ticks')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution (mean={residuals.mean():.1f}, std={residuals.std():.1f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot2_residuals.png'), dpi=150)
    plt.close()

    # ---- Plot 3: Binned calibration ----
    if best['bin_data']:
        fig, ax = plt.subplots(figsize=(8, 6))
        pred_means = [b['pred_mean'] for b in best['bin_data']]
        actual_means = [b['actual_mean'] for b in best['bin_data']]
        counts = [b['n'] for b in best['bin_data']]
        ax.scatter(pred_means, actual_means, s=[max(20, c * 0.5) for c in counts],
                   color='#EE7733', alpha=0.8, zorder=3)
        for pm, am, b in zip(pred_means, actual_means, best['bin_data']):
            ax.annotate(f"n={b['n']}", (pm, am), fontsize=7,
                        textcoords='offset points', xytext=(5, 5))
        lims = [min(min(pred_means), min(actual_means)) - 2,
                max(max(pred_means), max(actual_means)) + 2]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')
        ax.set_xlabel('Predicted mean (ticks)')
        ax.set_ylabel('Actual mean (ticks)')
        ax.set_title(f'Binned Calibration (slope={best["cal_slope"]:.2f})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'plot3_calibration.png'), dpi=150)
        plt.close()

    # ---- Plot 4: Model comparison (R² + direction acc) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = [r['label'] for r in valid]
    r2s = [r['r2'] for r in valid]
    dir_accs = [r['direction_acc'] * 100 for r in valid]

    colors = ['#EE7733' if r['model_type'] == 'gbm' else '#4477AA' for r in valid]
    ax1.bar(range(len(labels)), r2s, color=colors, alpha=0.85)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('R² (OOS)')
    ax1.set_title('OOS R² by Variant')
    ax1.axhline(0.10, color='red', linestyle='--', alpha=0.5, label='KILL threshold')
    ax1.axhline(0.30, color='green', linestyle='--', alpha=0.5, label='PROMOTE threshold')
    ax1.legend(fontsize=8)

    ax2.bar(range(len(labels)), dir_accs, color=colors, alpha=0.85)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('Direction Accuracy %')
    ax2.set_title('Direction from Sign(predicted)')
    ax2.axhline(52, color='red', linestyle='--', alpha=0.5, label='KILL threshold')
    ax2.axhline(58, color='green', linestyle='--', alpha=0.5, label='PROMOTE threshold')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot4_model_comparison.png'), dpi=150)
    plt.close()

    # ---- Plot 5: Feature importance (top 20) ----
    if best['feat_imp'] is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        imp = best['feat_imp']
        # Build feature names for 192D = 12 TFs × 16 features
        from tools.research.data import TF_HIERARCHY, FEATURE_NAMES
        feat_names = []
        for tf in TF_HIERARCHY:
            for fn in FEATURE_NAMES:
                feat_names.append(f'{tf}_{fn}')
        if len(feat_names) != len(imp):
            feat_names = [f'f{i}' for i in range(len(imp))]

        top_idx = np.argsort(imp)[-20:]
        ax.barh(range(20), imp[top_idx], color='#228833', alpha=0.85)
        ax.set_yticks(range(20))
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top 20 Features ({best["label"]})')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'plot5_feature_importance.png'), dpi=150)
        plt.close()

    print(f"  Plots saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Analysis CC: E[PnL] Tick Prediction')
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
    print(f'Analysis CC: E[PnL] Tick Prediction & Calibration')
    print(f'Data: {args.data}, Base TF: {args.base_tf}')
    print('=' * 70)

    # ---- 1. Load data & compute physics ----
    print("\n[1/4] Loading ATLAS data and computing physics...")
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

    # ---- 2. Build 192D matrices ----
    print("\n[2/4] Building stacked state matrices...")
    matrices, mfes, maes, meta = build_stacked_matrices(
        all_tf_states, args.base_tf, base_df,
        context_days=args.context_days, analysis_days=args.analysis_days,
    )
    if not matrices:
        print("ERROR: No matrices built.")
        return

    X_all = np.array([m.flatten() for m in matrices])  # (N, 192)
    print(f"  State matrix: {X_all.shape}")

    # ---- 3. Compute targets for each forward window ----
    print("\n[3/4] Computing forward tick targets...")
    ts_col = base_df['timestamp'].values
    ts_to_idx_df = {int(ts_col[i]): i for i in range(len(ts_col))}

    windows = [8, 16, 32]
    all_results = []

    for fwd_window in windows:
        print(f"\n  --- Forward window: {fwd_window} bars ---")
        y_raw = compute_signed_forward_ticks(base_df, fwd_window)

        # Map to matrix indices
        y_mapped = np.array([
            y_raw[ts_to_idx_df[m['ts']]] if m['ts'] in ts_to_idx_df else np.nan
            for m in meta
        ])

        valid = ~np.isnan(y_mapped)
        X_valid = X_all[valid]
        y_valid = y_mapped[valid]

        if len(X_valid) < 150:
            print(f"    Skipped: only {len(X_valid)} valid samples")
            continue

        print(f"    Samples: {len(X_valid)}, target mean={y_valid.mean():.1f}, "
              f"std={y_valid.std():.1f}")
        print(f"    LONG: {(y_valid > 0).sum()}, SHORT: {(y_valid < 0).sum()}")

        # GBM
        result = run_regression_experiment(
            X_valid, y_valid, f'GBM_fw{fwd_window}', model_type='gbm')
        all_results.append(result)

        # Ridge baseline
        result_ridge = run_regression_experiment(
            X_valid, y_valid, f'Ridge_fw{fwd_window}', model_type='ridge')
        all_results.append(result_ridge)

    # ---- 4. Compile results ----
    print("\n[4/4] Compiling results...")
    valid_results = [r for r in all_results if r is not None]

    lines = []
    lines.append("=" * 90)
    lines.append("ANALYSIS CC: E[PnL] TICK PREDICTION & CALIBRATION")
    lines.append(f"Data: {args.data}, Base TF: {args.base_tf}")
    lines.append("=" * 90)
    lines.append("")

    # Summary table
    header = (f"{'Variant':<20} {'N_test':>7} {'R²':>8} {'MAE':>8} "
              f"{'DirAcc':>8} {'CalSlope':>9} {'R²_base':>8}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in valid_results:
        cal_str = f"{r['cal_slope']:.2f}" if not np.isnan(r['cal_slope']) else "N/A"
        lines.append(f"{r['label']:<20} {r['n_test']:>7} {r['r2']:>8.4f} "
                     f"{r['mae']:>8.1f} {r['direction_acc']*100:>7.1f}% "
                     f"{cal_str:>9} {r['r2_baseline']:>8.4f}")

    # Binned calibration for best GBM
    gbm_results = [r for r in valid_results if r['model_type'] == 'gbm']
    if gbm_results:
        best_gbm = max(gbm_results, key=lambda r: r['r2'])
        lines.append("")
        lines.append(f"--- BINNED CALIBRATION ({best_gbm['label']}) ---")
        lines.append(f"  {'Bin':<15} {'N':>6} {'Pred Mean':>10} {'Actual Mean':>12}")
        for b in best_gbm['bin_data']:
            lines.append(f"  {b['bin']:<15} {b['n']:>6} {b['pred_mean']:>10.1f} "
                         f"{b['actual_mean']:>12.1f}")

        # Threshold gating
        lines.append("")
        lines.append(f"--- THRESHOLD GATING ({best_gbm['label']}) ---")
        lines.append(f"  {'Threshold':<20} {'N':>6} {'WinRate':>8} {'Avg Actual':>11}")
        for t in best_gbm['threshold_data']:
            lines.append(f"  {t['threshold']:<20} {t['n']:>6} {t['win_rate']*100:>7.1f}% "
                         f"{t['avg_actual']:>11.1f}")

    # Gate evaluation
    lines.append("")
    lines.append("=" * 70)
    lines.append("GATE EVALUATION")
    lines.append("=" * 70)

    if gbm_results:
        best = max(gbm_results, key=lambda r: r['r2'])
        lines.append(f"  Best model: {best['label']}")
        lines.append(f"  OOS R²:             {best['r2']:.4f}  (PROMOTE >= 0.30, KILL < 0.10)")
        lines.append(f"  Direction accuracy:  {best['direction_acc']*100:.1f}%  (PROMOTE >= 58%, KILL < 52%)")
        cal_str = f"{best['cal_slope']:.2f}" if not np.isnan(best['cal_slope']) else "N/A"
        lines.append(f"  Calibration slope:   {cal_str}  (PROMOTE 0.5-1.5)")
        lines.append(f"  MAE:                 {best['mae']:.1f} ticks")
        lines.append("")

        promote = (best['r2'] >= 0.30 and
                   best['direction_acc'] >= 0.58 and
                   not np.isnan(best['cal_slope']) and
                   0.5 <= best['cal_slope'] <= 1.5)
        kill = (best['r2'] < 0.10 or best['direction_acc'] < 0.52)

        if promote:
            verdict = ">>> PROMOTE: E[PnL] prediction has meaningful power — wire into pipeline"
        elif kill:
            verdict = ">>> KILL: Prediction too weak — regression on 192D physics insufficient"
        else:
            verdict = ">>> DEFER: Partial signal — explore richer features or ensemble methods"

        lines.append(verdict)

    results_text = "\n".join(lines)
    print("\n" + results_text)

    with open(os.path.join(OUT_DIR, 'results.txt'), 'w') as f:
        f.write(results_text)

    # Plots
    print("\n  Generating plots...")
    make_plots(all_results)

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

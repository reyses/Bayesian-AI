"""
Screening Plots — Minitab-style DOE visualizations
===================================================
Generates: Pareto chart, Heatmap, Stepwise R2, Factor effects.

Usage:
    python tools/screening_plots.py

Reads from checkpoint, generates plots to tools/plots/
"""

import sys, os
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import screening functions
from tools.waveform_screening import (
    load_templates, extract_matrices, pad_to_fixed_depth,
    flatten_matrices, screen_factors, FEATURE_NAMES, TF_LABELS
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')


def plot_pareto(results, save_path, top_n=30):
    """Minitab-style Pareto chart of factor effects."""
    names = [r[0].replace('__', '\n') for r in results[:top_n]]
    abs_corr = [r[2] for r in results[:top_n]]
    signs = [r[1] for r in results[:top_n]]

    # Colors: blue=positive correlation, red=negative
    colors = ['#2196F3' if s > 0 else '#F44336' for s in signs]

    fig, ax1 = plt.subplots(figsize=(16, 8))

    bars = ax1.bar(range(top_n), abs_corr, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('|Correlation| with MFE', fontsize=12)
    ax1.set_xlabel('Factor (depth__feature)', fontsize=12)
    ax1.set_title('PARETO CHART: Factor Screening (16Fx12TF -> MFE)\n'
                   'Blue = positive correlation, Red = negative', fontsize=14)
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=7)

    # Cumulative line
    cumsum = np.cumsum(abs_corr) / sum(abs_corr) * 100
    ax2 = ax1.twinx()
    ax2.plot(range(top_n), cumsum, 'k-o', markersize=3, linewidth=1.5)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax2.set_ylim(0, 105)

    # 80% line
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    ax2.text(top_n - 1, 82, '80%', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pareto chart saved: {save_path}")


def plot_heatmap(results, save_path):
    """Heatmap: Features (x) vs Timeframe Depth (y), color = correlation."""
    # Build matrix
    n_tf = len(TF_LABELS)
    n_feat = len(FEATURE_NAMES)
    matrix = np.zeros((n_tf, n_feat))

    result_dict = {r[0]: r[1] for r in results}
    for d in range(n_tf):
        tf_lbl = TF_LABELS[d]
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f]
            key = f"{tf_lbl}__{f_lbl}"
            matrix[d, f] = result_dict.get(key, 0.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_tf))
    ax.set_yticklabels(TF_LABELS, fontsize=9)

    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Timeframe Depth', fontsize=12)
    ax.set_title('CORRELATION HEATMAP: 16F x 12TF -> MFE\n'
                  'Red = positive (higher feature -> higher MFE), '
                  'Blue = negative', fontsize=13)

    # Add text annotations
    for d in range(n_tf):
        for f in range(n_feat):
            val = matrix[d, f]
            if abs(val) > 0.05:
                color = 'white' if abs(val) > 0.15 else 'black'
                ax.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation with MFE', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {save_path}")


def plot_stepwise_r2(flat, col_names, mfes, save_path, top_k=30):
    """Stepwise R2 plot: cumulative adj-R2 as factors are added."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    screening = screen_factors(flat, col_names, mfes)
    top_names = [s[0] for s in screening[:top_k]]
    top_indices = [col_names.index(n) for n in top_names]

    scaler = StandardScaler()
    r2_vals = []
    adj_r2_vals = []

    for step in range(1, top_k + 1):
        selected = top_indices[:step]
        X = scaler.fit_transform(flat[:, selected])
        reg = LinearRegression().fit(X, mfes)
        r2 = reg.score(X, mfes)
        n, k = X.shape
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)
        r2_vals.append(r2)
        adj_r2_vals.append(adj_r2)

    fig, ax = plt.subplots(figsize=(14, 6))
    steps = range(1, top_k + 1)

    ax.plot(steps, r2_vals, 'b-o', markersize=4, label='R2', linewidth=2)
    ax.plot(steps, adj_r2_vals, 'r-s', markersize=4, label='adj-R2', linewidth=2)
    ax.fill_between(steps, adj_r2_vals, alpha=0.1, color='red')

    # Mark diminishing returns
    deltas = np.diff(r2_vals)
    elbow = np.argmax(deltas < 0.005) + 1 if any(d < 0.005 for d in deltas) else top_k
    ax.axvline(x=elbow, color='gray', linestyle='--', alpha=0.5)
    ax.text(elbow + 0.5, max(r2_vals) * 0.95, f'Elbow @ {elbow} factors',
            fontsize=9, color='gray')

    ax.set_xlabel('Number of Factors', fontsize=12)
    ax.set_ylabel('R2 / adj-R2', fontsize=12)
    ax.set_title('STEPWISE REGRESSION: Cumulative R2 as factors added\n'
                  f'Max adj-R2 = {max(adj_r2_vals):.4f} with {top_k} factors', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # X-axis labels
    short_names = [n.replace('__', '\n') for n in top_names]
    ax.set_xticks(steps)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Stepwise R2 plot saved: {save_path}")


def plot_depth_importance(results, save_path):
    """Bar chart: mean |corr| by timeframe depth — which zoom level matters?"""
    depths = []
    means = []
    maxes = []

    for d in range(12):
        prefix = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        depth_factors = [a for n, c, a in results if n.startswith(prefix + '__')]
        if depth_factors:
            depths.append(prefix)
            means.append(np.mean(depth_factors))
            maxes.append(max(depth_factors))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(depths))
    width = 0.35

    ax.bar([i - width/2 for i in x], means, width, label='Mean |corr|',
           color='#2196F3', edgecolor='white')
    ax.bar([i + width/2 for i in x], maxes, width, label='Max |corr|',
           color='#FF9800', edgecolor='white')

    ax.set_xlabel('Timeframe Depth (macro -> micro)', fontsize=12)
    ax.set_ylabel('|Correlation| with MFE', fontsize=12)
    ax.set_title('DEPTH IMPORTANCE: Which timeframe level predicts MFE?\n'
                  'Macro (d0) = 4h, Micro (d11) = 15s', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Depth importance saved: {save_path}")


def plot_feature_importance(results, save_path):
    """Bar chart: mean |corr| by feature — which feature matters most?"""
    features = []
    means = []
    maxes = []
    best_depths = []

    for f_name in FEATURE_NAMES:
        feat_factors = [(n, c, a) for n, c, a in results if n.endswith(f'__{f_name}')]
        if feat_factors:
            abs_corrs = [a for _, _, a in feat_factors]
            best_idx = np.argmax(abs_corrs)
            features.append(f_name)
            means.append(np.mean(abs_corrs))
            maxes.append(max(abs_corrs))
            best_depths.append(feat_factors[best_idx][0].split('__')[0])

    # Sort by mean
    order = np.argsort(means)[::-1]
    features = [features[i] for i in order]
    means = [means[i] for i in order]
    maxes = [maxes[i] for i in order]
    best_depths = [best_depths[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(features))

    bars = ax.bar(x, maxes, color='#FF9800', edgecolor='white', alpha=0.7, label='Max |corr|')
    ax.bar(x, means, color='#2196F3', edgecolor='white', label='Mean |corr|')

    # Annotate best depth
    for i, (mx, bd) in enumerate(zip(maxes, best_depths)):
        ax.text(i, mx + 0.005, bd, ha='center', va='bottom', fontsize=7,
                color='#666', rotation=45)

    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('|Correlation| with MFE', fontsize=12)
    ax.set_title('FEATURE IMPORTANCE: Which features predict MFE?\n'
                  'Labels show best timeframe depth for each feature', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature importance saved: {save_path}")


def plot_main_effects(flat, col_names, mfes, results, save_path, top_n=6):
    """Minitab-style main effects plot for top factors."""
    top_factors = results[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (name, corr, abs_corr) in enumerate(top_factors):
        ax = axes[i]
        col_idx = col_names.index(name)
        x = flat[:, col_idx]

        # Bin into quartiles
        quartiles = np.percentile(x[x != 0], [25, 50, 75])
        bins = [-np.inf] + list(quartiles) + [np.inf]
        labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)']

        bin_means = []
        bin_stds = []
        for b in range(len(bins) - 1):
            mask = (x >= bins[b]) & (x < bins[b + 1])
            if mask.any():
                bin_means.append(np.mean(mfes[mask]))
                bin_stds.append(np.std(mfes[mask]) / np.sqrt(mask.sum()))
            else:
                bin_means.append(0)
                bin_stds.append(0)

        color = '#2196F3' if corr > 0 else '#F44336'
        ax.bar(range(4), bin_means, yerr=bin_stds, color=color, alpha=0.7,
               edgecolor='white', capsize=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f'{name}\ncorr={corr:+.3f}', fontsize=10)
        ax.set_ylabel('Mean MFE', fontsize=9)
        ax.axhline(y=np.mean(mfes), color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('MAIN EFFECTS PLOT: Top 6 Factors\n'
                 'Bars show mean MFE by factor quartile, error bars = SE',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Main effects plot saved: {save_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading data...")
    templates = load_templates()
    matrices, mfes, maes, meta = extract_matrices(templates)
    padded = pad_to_fixed_depth(matrices, max_depth=12)
    flat, col_names = flatten_matrices(padded)
    results = screen_factors(flat, col_names, mfes)

    print(f"\nGenerating Minitab-style plots...")

    # 1. Pareto
    plot_pareto(results, os.path.join(PLOTS_DIR, '1_pareto_chart.png'))

    # 2. Heatmap
    plot_heatmap(results, os.path.join(PLOTS_DIR, '2_correlation_heatmap.png'))

    # 3. Stepwise R2
    plot_stepwise_r2(flat, col_names, mfes,
                     os.path.join(PLOTS_DIR, '3_stepwise_r2.png'))

    # 4. Depth importance
    plot_depth_importance(results, os.path.join(PLOTS_DIR, '4_depth_importance.png'))

    # 5. Feature importance
    plot_feature_importance(results, os.path.join(PLOTS_DIR, '5_feature_importance.png'))

    # 6. Main effects
    plot_main_effects(flat, col_names, mfes, results,
                      os.path.join(PLOTS_DIR, '6_main_effects.png'))

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    main()

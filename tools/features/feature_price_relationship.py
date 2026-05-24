"""Feature-Price Relationship — individual delta feature vs price movement.

For each of the 12 PhysicsEngine features, plots:
1. Scatter: feature delta vs next-bar price change (ticks)
2. Binned: mean price change per feature delta decile
3. Correlation stats: Spearman, Pearson, mutual information

Outputs:
  - tools/plots/feature_price/ — one PNG per feature (12 total)
  - reports/findings/feature_price_summary.txt — correlation table

Usage:
    python tools/feature_price_relationship.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import StatisticalFieldEngine

TICK = 0.25

# 12 PhysicsEngine features + display names
FEATURES = [
    ('F_momentum',                    'F_momentum (fm)'),
    ('z_score',                       'Z-Score (z)'),
    ('dmi_plus',                      'DMI+ (dmi_p)'),
    ('dmi_minus',                     'DMI- (dmi_m)'),
    ('adx_strength',                  'ADX (adx)'),
    ('velocity',                      'Velocity (vel)'),
    ('volume_delta',                  'Volume Delta (vol)'),
    ('hurst_exponent',                'Hurst (hurst)'),
    ('P_at_center',                   'P_center'),
    ('oscillation_entropy_normalized','Coherence (coh)'),
    ('regression_sigma',              'Sigma'),
    ('term_pid',                      'PID'),
]

# Lookahead windows to test (bars into the future)
LOOKAHEADS = [1, 3, 5, 10]


def load_oos_states(data_dir='DATA/ATLAS_OOS', max_bars=0):
    """Load 1m data and compute states."""
    print(f'Loading 1m data from {data_dir}...')
    oos_files = sorted(glob.glob(os.path.join(data_dir, '1m', '*.parquet')))
    if not oos_files:
        print(f'ERROR: No 1m parquet files found in {data_dir}/1m/')
        sys.exit(1)
    oos = pd.concat([pd.read_parquet(f) for f in oos_files], ignore_index=True)
    if max_bars > 0 and len(oos) > max_bars:
        oos = oos.iloc[:max_bars]
    print(f'  {len(oos):,} bars from {len(oos_files)} files')

    engine = StatisticalFieldEngine()
    print('Computing states...')
    raw = engine.batch_compute_states(oos)
    states = []
    for r in raw:
        if r and isinstance(r, dict) and 'state' in r:
            states.append(r['state'])
        else:
            states.append(None)

    closes = oos['close'].values
    timestamps = oos['timestamp'].values
    print(f'  {sum(1 for s in states if s is not None):,} valid states')
    return states, closes, timestamps


def extract_features_and_deltas(states, closes):
    """Extract per-bar feature values, deltas, and future price changes."""
    n = len(states)
    max_la = max(LOOKAHEADS)

    # Pre-extract all feature values
    feat_vals = {attr: np.full(n, np.nan) for attr, _ in FEATURES}
    for i, s in enumerate(states):
        if s is None:
            continue
        for attr, _ in FEATURES:
            feat_vals[attr][i] = getattr(s, attr, np.nan)

    # Compute deltas (bar-over-bar change)
    feat_deltas = {}
    for attr, _ in FEATURES:
        v = feat_vals[attr]
        delta = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(v[i]) and not np.isnan(v[i-1]):
                delta[i] = v[i] - v[i-1]
        feat_deltas[attr] = delta

    # Future price changes (in ticks)
    price_changes = {}
    for la in LOOKAHEADS:
        pc = np.full(n, np.nan)
        for i in range(n - la):
            pc[i] = (closes[i + la] - closes[i]) / TICK
        price_changes[la] = pc

    return feat_vals, feat_deltas, price_changes


def plot_feature(attr, label, deltas, price_changes, out_dir):
    """Plot one feature's delta vs price change."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(LOOKAHEADS), figsize=(6 * len(LOOKAHEADS), 12))
    fig.suptitle(f'{label} — Delta vs Future Price Change', fontsize=16, fontweight='bold')

    results = {}

    for col, la in enumerate(LOOKAHEADS):
        d = deltas
        pc = price_changes[la]

        # Valid mask
        mask = ~np.isnan(d) & ~np.isnan(pc)
        x = d[mask]
        y = pc[mask]

        if len(x) < 100:
            results[la] = {'n': len(x), 'spearman': 0, 'pearson': 0}
            continue

        # Stats
        spear_r, spear_p = sp_stats.spearmanr(x, y)
        pear_r, pear_p = sp_stats.pearsonr(x, y)
        results[la] = {
            'n': len(x),
            'spearman': spear_r,
            'spearman_p': spear_p,
            'pearson': pear_r,
            'pearson_p': pear_p,
        }

        # Row 1: Scatter (subsampled for speed)
        ax = axes[0, col]
        idx = np.random.choice(len(x), min(5000, len(x)), replace=False)
        ax.scatter(x[idx], y[idx], alpha=0.1, s=2, c='steelblue')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel(f'delta({label})')
        ax.set_ylabel(f'price change +{la} bars (ticks)')
        ax.set_title(f'+{la} bar | r_s={spear_r:.3f} r_p={pear_r:.3f}')

        # Clip outliers for readability
        x_lo, x_hi = np.percentile(x, [1, 99])
        y_lo, y_hi = np.percentile(y, [1, 99])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        # Row 2: Binned mean (deciles)
        ax2 = axes[1, col]
        try:
            bins = pd.qcut(x, 10, duplicates='drop')
            df_tmp = pd.DataFrame({'x': x, 'y': y, 'bin': bins})
            grouped = df_tmp.groupby('bin', observed=True)['y']
            means = grouped.mean()
            sems = grouped.sem()
            counts = grouped.count()

            bin_centers = np.arange(len(means))
            ax2.bar(bin_centers, means.values, yerr=sems.values,
                    color='steelblue', alpha=0.7, capsize=3)
            ax2.axhline(0, color='red', lw=1, ls='--')
            ax2.set_xticks(bin_centers)
            ax2.set_xticklabels([f'{iv.left:.2g}' for iv in means.index],
                                rotation=45, ha='right', fontsize=7)
            ax2.set_xlabel(f'delta({label}) decile')
            ax2.set_ylabel(f'mean price change +{la} (ticks)')
            ax2.set_title(f'Binned mean (n={len(x):,})')

            # Annotate counts
            for j, c in enumerate(counts.values):
                ax2.text(j, 0, f'n={c}', ha='center', va='bottom', fontsize=6, alpha=0.5)
        except Exception:
            ax2.text(0.5, 0.5, 'binning failed', transform=ax2.transAxes, ha='center')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{attr}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  {label}: saved {path}')
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Feature-price relationship analysis')
    parser.add_argument('--data', default='DATA/ATLAS_OOS',
                        help='Data directory (default: DATA/ATLAS_OOS)')
    parser.add_argument('--max-bars', type=int, default=0,
                        help='Limit bars (0=all, 1440=~1 day of 1m)')
    args = parser.parse_args()

    out_dir = os.path.join('tools', 'plots', 'feature_price')
    os.makedirs(out_dir, exist_ok=True)

    states, closes, timestamps = load_oos_states(args.data, args.max_bars)
    feat_vals, feat_deltas, price_changes = extract_features_and_deltas(states, closes)

    print(f'\nPlotting {len(FEATURES)} features x {len(LOOKAHEADS)} lookaheads...')
    all_results = {}
    for attr, label in FEATURES:
        all_results[attr] = plot_feature(
            attr, label, feat_deltas[attr], price_changes, out_dir)

    # Summary table
    summary_path = os.path.join('reports', 'findings', 'feature_price_summary.txt')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    lines = []
    lines.append('=' * 80)
    lines.append('FEATURE-PRICE RELATIONSHIP SUMMARY')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Data: OOS 1m ({sum(1 for s in states if s is not None):,} bars)')
    lines.append(f'Lookaheads: {LOOKAHEADS} bars')
    lines.append('=' * 80)
    lines.append('')

    # Spearman correlation table
    lines.append(f'{"Feature":<30}' + ''.join(f'  +{la}bar r_s' for la in LOOKAHEADS)
                 + ''.join(f'  +{la}bar r_p' for la in LOOKAHEADS))
    lines.append('-' * (30 + 12 * len(LOOKAHEADS) * 2))

    for attr, label in FEATURES:
        r = all_results[attr]
        parts = [f'{label:<30}']
        for la in LOOKAHEADS:
            s = r.get(la, {})
            parts.append(f'  {s.get("spearman", 0):+.4f}  ')
        for la in LOOKAHEADS:
            s = r.get(la, {})
            parts.append(f'  {s.get("pearson", 0):+.4f}  ')
        lines.append(''.join(parts))

    lines.append('')
    lines.append('r_s = Spearman rank correlation (monotonic relationship)')
    lines.append('r_p = Pearson correlation (linear relationship)')
    lines.append('Positive r = feature delta predicts price increase')
    lines.append('Negative r = feature delta predicts price decrease')
    lines.append('')

    # Rank by absolute Spearman at +1 bar
    lines.append('RANKED BY |SPEARMAN| AT +1 BAR:')
    ranked = sorted(all_results.items(),
                    key=lambda x: abs(x[1].get(1, {}).get('spearman', 0)),
                    reverse=True)
    for i, (attr, r) in enumerate(ranked, 1):
        label = dict(FEATURES)[attr]
        s1 = r.get(1, {}).get('spearman', 0)
        s10 = r.get(10, {}).get('spearman', 0)
        lines.append(f'  {i:>2}. {label:<30} +1bar={s1:+.4f}  +10bar={s10:+.4f}')

    lines.append('')
    lines.append(f'Plots saved: {out_dir}/')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'\nSummary: {summary_path}')
    print(f'Plots:   {out_dir}/')

    # Print quick ranking
    print('\nQuick ranking (|Spearman| at +1 bar):')
    for i, (attr, r) in enumerate(ranked[:5], 1):
        label = dict(FEATURES)[attr]
        s = r.get(1, {}).get('spearman', 0)
        print(f'  {i}. {label}: {s:+.4f}')


if __name__ == '__main__':
    main()

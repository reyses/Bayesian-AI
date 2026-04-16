"""
Feature Response Surface — find which features predict trade outcomes.

Runs the baseline engine on IS data, captures every trade with entry features,
then analyzes which features (and combinations) separate winners from losers.

Methods:
1. Univariate: Welch t-test winners vs losers per feature
2. Importance: gradient boosting on binary outcome
3. Top interactions: 2D win-rate heatmaps for top features

Output: reports/findings/feature_response_surface.txt + .csv

Usage:
    python tools/feature_response_surface.py
    python tools/feature_response_surface.py --tier RIDE_AGAINST  # one tier only
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine
from core.features import FEATURE_NAMES

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
OUTPUT_DIR = 'reports/findings'


def collect_trades(tier_filter=None, max_days=None):
    """Run engine on IS, return DataFrame: features + outcome."""
    feat_files = sorted(f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                        if '2025_' in os.path.basename(f))
    if max_days:
        feat_files = feat_files[:max_days]

    print(f'Processing {len(feat_files)} IS days...')
    engine = BlendedEngine(use_cnn=False)
    rows = []

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades:
            tier = t.get('entry_tier', '?')
            if tier_filter and tier != tier_filter:
                continue
            entry_79d = t.get('entry_79d', None)
            if entry_79d is None:
                continue
            if isinstance(entry_79d, list):
                entry_79d = np.array(entry_79d)
            if len(entry_79d) < 91:
                continue

            row = {f: float(entry_79d[i]) for i, f in enumerate(FEATURE_NAMES[:91])}
            row['_tier'] = tier
            row['_pnl'] = t['pnl']
            row['_held'] = t.get('held', 0)
            row['_win'] = 1 if t['pnl'] > 0 else 0
            row['_day'] = day
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f'Captured {len(df)} trades')
    return df


def univariate_analysis(df):
    """For each feature: compare distributions of winners vs losers."""
    from scipy import stats

    winners = df[df['_win'] == 1]
    losers = df[df['_win'] == 0]

    results = []
    for col in FEATURE_NAMES[:91]:
        if col not in df.columns:
            continue
        w_vals = winners[col].dropna().values
        l_vals = losers[col].dropna().values
        if len(w_vals) < 10 or len(l_vals) < 10:
            continue

        w_mean, l_mean = w_vals.mean(), l_vals.mean()
        w_std, l_std = w_vals.std(), l_vals.std()

        # Welch t-test
        try:
            t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)
        except Exception:
            t_stat, p_val = 0, 1.0

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((w_std**2 + l_std**2) / 2)
        cohen_d = (w_mean - l_mean) / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': col,
            'win_mean': w_mean, 'win_std': w_std,
            'loss_mean': l_mean, 'loss_std': l_std,
            't_stat': t_stat, 'p_value': p_val,
            'cohen_d': cohen_d,
            'abs_d': abs(cohen_d),
        })

    return pd.DataFrame(results).sort_values('abs_d', ascending=False)


def importance_analysis(df):
    """Gradient boosting to find feature importance + non-linear interactions."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        print('sklearn not available — skipping importance analysis')
        return None

    X = df[[c for c in FEATURE_NAMES[:91] if c in df.columns]].fillna(0)
    y = df['_win'].values

    if len(X) < 100:
        return None

    print(f'Training gradient boosting on {len(X)} trades...')
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                        learning_rate=0.1, random_state=42)
    model.fit(X, y)

    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    return importances


def per_tier_analysis(df):
    """Univariate per tier — different tiers may have different drivers."""
    tier_results = {}
    for tier in df['_tier'].unique():
        sub = df[df['_tier'] == tier]
        if len(sub) < 50:
            continue
        wr = sub['_win'].mean() * 100
        tier_results[tier] = {
            'n': len(sub),
            'wr': wr,
            'avg_pnl': sub['_pnl'].mean(),
            'top_features': univariate_analysis(sub).head(10),
        }
    return tier_results


def write_report(df, uni, imp, tier_results, out_path):
    lines = []
    lines.append('=' * 80)
    lines.append('FEATURE RESPONSE SURFACE — IS dataset')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append('=' * 80)
    lines.append('')

    lines.append(f'Trades analyzed: {len(df):,}')
    lines.append(f'Winners: {(df["_win"]==1).sum():,} ({df["_win"].mean()*100:.0f}%)')
    lines.append(f'Losers: {(df["_win"]==0).sum():,}')
    lines.append('')

    # Top 20 features by Cohen's d
    lines.append('TOP 20 FEATURES BY EFFECT SIZE (winners vs losers, |Cohen d|):')
    lines.append(f'{"Feature":<25} {"Cohen d":>10} {"p-value":>10} {"Win mean":>12} {"Loss mean":>12}')
    lines.append('-' * 75)
    for _, r in uni.head(20).iterrows():
        sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else ''
        lines.append(f'{r["feature"]:<25} {r["cohen_d"]:>+10.3f} {r["p_value"]:>10.4f} '
                     f'{r["win_mean"]:>12.4f} {r["loss_mean"]:>12.4f} {sig}')
    lines.append('')

    if imp is not None:
        lines.append('TOP 20 FEATURES BY GRADIENT BOOSTING IMPORTANCE:')
        lines.append(f'{"Feature":<25} {"Importance":>12}')
        lines.append('-' * 40)
        for _, r in imp.head(20).iterrows():
            lines.append(f'{r["feature"]:<25} {r["importance"]:>12.4f}')
        lines.append('')

    lines.append('=' * 80)
    lines.append('PER-TIER FEATURE DRIVERS')
    lines.append('=' * 80)
    for tier, info in sorted(tier_results.items(), key=lambda x: -x[1]['n']):
        lines.append(f'\n{tier}: {info["n"]} trades, WR={info["wr"]:.0f}%, avg=${info["avg_pnl"]:+.1f}')
        lines.append(f'  Top features:')
        for _, r in info['top_features'].head(5).iterrows():
            sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else ''
            lines.append(f'    {r["feature"]:<25} d={r["cohen_d"]:>+6.2f} '
                         f'p={r["p_value"]:.4f} {sig}')

    lines.append('')
    lines.append('=' * 80)
    lines.append('UNUSED FEATURES WITH SIGNAL (not in current physics):')
    lines.append('=' * 80)
    USED_FEATURES = {
        '1m_z_se', '1m_variance_ratio', '1m_velocity', '1m_acceleration',
        '1m_p_at_center', '1m_vol_rel', '1m_hurst', '1m_bar_range',
        '1m_dmi_diff', '5m_wick_ratio', '15m_wick_ratio',
        '5m_velocity', '5m_acceleration', '5m_z_se', '15m_z_se',
        '1h_z_se', '1h_velocity',
    }
    unused_signals = uni[~uni['feature'].isin(USED_FEATURES)]
    unused_with_signal = unused_signals[unused_signals['p_value'] < 0.01].head(15)
    if len(unused_with_signal) > 0:
        lines.append(f'{"Feature":<25} {"Cohen d":>10} {"p-value":>10}')
        lines.append('-' * 50)
        for _, r in unused_with_signal.iterrows():
            lines.append(f'{r["feature"]:<25} {r["cohen_d"]:>+10.3f} {r["p_value"]:>10.4f}')
    else:
        lines.append('  None found (all signal features already in physics)')

    lines.append('')
    lines.append('=' * 80)

    report = '\n'.join(lines)
    print(report)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=str, default=None,
                        help='Analyze only one tier')
    parser.add_argument('--days', type=int, default=None,
                        help='Limit to first N days (for speed)')
    args = parser.parse_args()

    df = collect_trades(tier_filter=args.tier, max_days=args.days)
    if len(df) < 100:
        print('Not enough trades collected.')
        return

    # Save raw trades for downstream analysis
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUTPUT_DIR, 'feature_response_trades.parquet')
    df.to_parquet(raw_path, index=False)
    print(f'Raw trades saved: {raw_path}')

    print('\nUnivariate analysis...')
    uni = univariate_analysis(df)

    print('Importance analysis...')
    imp = importance_analysis(df)

    print('Per-tier analysis...')
    tier_results = per_tier_analysis(df)

    suffix = f'_{args.tier}' if args.tier else ''
    out_path = os.path.join(OUTPUT_DIR, f'feature_response_surface{suffix}.txt')
    write_report(df, uni, imp, tier_results, out_path)


if __name__ == '__main__':
    main()

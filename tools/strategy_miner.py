"""
Strategy Miner — data-driven entry strategy discovery.

Loads 1m ATLAS bars, computes 13D features + volume + time-of-day,
then scans feature thresholds and combinations to find profitable
entry conditions at multiple forward horizons.

Outputs:
  reports/findings/strategy_miner_YYYYMMDD.md  — ranked strategy proposals
  reports/findings/strategy_miner_YYYYMMDD.csv — raw scan results

Each proposed strategy includes:
  - Entry conditions (feature thresholds)
  - Direction (LONG/SHORT)
  - Optimal exit horizon
  - WR, avg PnL, Sharpe, total PnL
  - Trade count (must be > 30 for statistical validity)

Usage:
  python tools/strategy_miner.py                   # IS data (2025)
  python tools/strategy_miner.py --oos              # OOS data (2026)
  python tools/strategy_miner.py --min-trades 50    # stricter filter
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import gc
import glob
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.statistical_field_engine import StatisticalFieldEngine
from training.train_trade_cnn import extract_features_13d as extract_13d_batch, FEATURE_NAMES_13D

TICK = 0.25
TV = 0.50  # $0.50 per tick MNQ

# Config
OOS = '--oos' in sys.argv
MIN_TRADES = 30
if '--min-trades' in sys.argv:
    idx = sys.argv.index('--min-trades')
    MIN_TRADES = int(sys.argv[idx + 1])

# Forward horizons to test (1m bars)
HORIZONS = [3, 5, 10, 15, 20]

# Feature thresholds to scan
# Each entry: (feature_name, feature_index, thresholds, direction_rule)
# direction_rule: 'above_long' means above threshold = LONG, below = SHORT
#                 'extreme_fade' means extreme = fade (go opposite)
SCAN_RULES = [
    # z_se: extreme = fade toward mean
    ('z_se', 5, [-4, -3, -2, 2, 3, 4], 'extreme_fade'),
    # variance_ratio: low = stable/mean-revert, high = trending
    ('vr', 8, [0.3, 0.5, 0.7, 1.0, 1.3], 'threshold'),
    # lambda (vr - 1 essentially): negative = mean-revert regime
    ('lam', None, [-0.7, -0.5, -0.3, 0.0, 0.3], 'threshold'),
    # dmi_diff: positive = bullish, negative = bearish
    ('dmi_diff', 0, [-5, -3, -1, 1, 3, 5], 'directional'),
    # vol_rel: volume relative to SMA
    ('vol_rel', 2, [0.5, 1.0, 1.5, 2.0, 3.0], 'threshold'),
    # price_accel: momentum
    ('price_accel', 6, None, 'directional'),
    # bar_range: volatility
    ('bar_range', 9, [10, 20, 40, 80], 'threshold'),
]


def load_1m_with_features(oos=False):
    """Load 1m ATLAS, compute SFE features, return enriched DataFrame."""
    atlas_dir = 'DATA/ATLAS/1m'
    files = sorted(glob.glob(os.path.join(atlas_dir, '*.parquet')))

    if oos:
        # OOS = 2026 data
        files = [f for f in files if '2026_' in os.path.basename(f)]
    else:
        # IS = 2025 data
        files = [f for f in files if '2025_' in os.path.basename(f)]

    print(f'Loading {len(files)} 1m files ({"OOS" if oos else "IS"})...')
    dfs = [pd.read_parquet(f) for f in tqdm(files, desc='  Loading')]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f'  {len(df):,} bars loaded')

    # Compute SFE features
    print('  Computing SFE states...')
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)

    print('  Extracting 13D features...')
    feats = extract_13d_batch(states, df)
    del states, sfe
    gc.collect()

    # Build enriched df
    for i, name in enumerate(FEATURE_NAMES_13D):
        df[name] = feats[:, i]

    # Add derived features
    df['hour'] = (df['timestamp'] % 86400) // 3600

    # 15-bar rolling trend
    df['trend_15'] = df['close'] - df['close'].shift(15)

    # Volume SMA for relative volume
    df['vol_sma30'] = df['volume'].rolling(30, min_periods=1).mean()
    df['vol_rel_live'] = df['volume'] / df['vol_sma30'].clip(lower=1)

    print(f'  Features computed: {len(df):,} bars x {len(df.columns)} columns')
    return df


def compute_forward_returns(df):
    """Compute forward price changes at multiple horizons."""
    print('  Computing forward returns...')
    for h in HORIZONS:
        # Forward close price
        df[f'fwd_{h}_close'] = df['close'].shift(-h)
        # Forward high (best long)
        df[f'fwd_{h}_high'] = df['high'].rolling(h).max().shift(-h)
        # Forward low (best short)
        df[f'fwd_{h}_low'] = df['low'].rolling(h).min().shift(-h)

        # PnL in ticks
        df[f'fwd_{h}_long'] = (df[f'fwd_{h}_close'] - df['close']) / TICK * TV
        df[f'fwd_{h}_short'] = (df['close'] - df[f'fwd_{h}_close']) / TICK * TV
        # Best case PnL
        df[f'fwd_{h}_long_best'] = (df[f'fwd_{h}_high'] - df['close']) / TICK * TV
        df[f'fwd_{h}_short_best'] = (df['close'] - df[f'fwd_{h}_low']) / TICK * TV

    return df


def scan_single_feature(df, feat_name, feat_col, thresholds, rule_type):
    """Scan a single feature at various thresholds."""
    results = []

    if thresholds is None:
        # Auto-generate from percentiles
        vals = df[feat_col].dropna()
        thresholds = [np.percentile(vals, p) for p in [10, 25, 50, 75, 90]]

    for thresh in thresholds:
        for horizon in HORIZONS:
            long_col = f'fwd_{horizon}_long'
            short_col = f'fwd_{horizon}_short'

            if rule_type == 'extreme_fade':
                # z > thresh: SHORT (fade), z < -thresh: LONG (fade)
                if thresh > 0:
                    mask = df[feat_col] > thresh
                    pnl_col = short_col
                    direction = 'SHORT'
                    label = f'{feat_name} > {thresh}'
                else:
                    mask = df[feat_col] < thresh
                    pnl_col = long_col
                    direction = 'LONG'
                    label = f'{feat_name} < {thresh}'

            elif rule_type == 'directional':
                # Positive = LONG, negative = SHORT
                if thresh > 0:
                    mask = df[feat_col] > thresh
                    pnl_col = long_col
                    direction = 'LONG'
                    label = f'{feat_name} > {thresh}'
                else:
                    mask = df[feat_col] < thresh
                    pnl_col = short_col
                    direction = 'SHORT'
                    label = f'{feat_name} < {thresh}'

            elif rule_type == 'threshold':
                # Above/below threshold, test both directions
                for dir_label, pnl_c, d in [('LONG', long_col, 'LONG'), ('SHORT', short_col, 'SHORT')]:
                    mask_above = df[feat_col] > thresh
                    mask_below = df[feat_col] < thresh

                    for m, cond_label in [(mask_above, f'{feat_name} > {thresh}'),
                                          (mask_below, f'{feat_name} < {thresh}')]:
                        sub = df.loc[m, pnl_c].dropna()
                        if len(sub) < MIN_TRADES:
                            continue
                        wr = (sub > 0).mean() * 100
                        avg = sub.mean()
                        total = sub.sum()
                        sharpe = sub.mean() / sub.std() if sub.std() > 0 else 0
                        results.append({
                            'strategy': f'{cond_label} → {d}',
                            'condition': cond_label,
                            'direction': d,
                            'horizon': horizon,
                            'n_trades': len(sub),
                            'wr': wr,
                            'avg_pnl': avg,
                            'total_pnl': total,
                            'sharpe': sharpe,
                            'median_pnl': sub.median(),
                        })
                continue  # skip the common path below

            else:
                continue

            if rule_type != 'threshold':
                sub = df.loc[mask, pnl_col].dropna()
                if len(sub) < MIN_TRADES:
                    continue
                wr = (sub > 0).mean() * 100
                avg = sub.mean()
                total = sub.sum()
                sharpe = sub.mean() / sub.std() if sub.std() > 0 else 0
                results.append({
                    'strategy': f'{label} → {direction}',
                    'condition': label,
                    'direction': direction,
                    'horizon': horizon,
                    'n_trades': len(sub),
                    'wr': wr,
                    'avg_pnl': avg,
                    'total_pnl': total,
                    'sharpe': sharpe,
                    'median_pnl': sub.median(),
                })

    return results


def scan_combinations(df):
    """Scan 2-feature AND combinations for the top single features."""
    print('  Scanning 2-feature combinations...')
    results = []

    # Define combo conditions to test (from Nightmare protocol knowledge)
    combos = [
        # Nightmare core: z extreme + stable regime
        ('z_se > 2 + vr < 0.7 → SHORT', lambda d: (d['z_se'] > 2) & (d['variance_ratio'] < 0.7), 'SHORT'),
        ('z_se < -2 + vr < 0.7 → LONG', lambda d: (d['z_se'] < -2) & (d['variance_ratio'] < 0.7), 'LONG'),
        ('z_se > 3 + vr < 0.5 → SHORT', lambda d: (d['z_se'] > 3) & (d['variance_ratio'] < 0.5), 'SHORT'),
        ('z_se < -3 + vr < 0.5 → LONG', lambda d: (d['z_se'] < -3) & (d['variance_ratio'] < 0.5), 'LONG'),

        # Trend + momentum
        ('trend > 20 + dmi > 3 → LONG', lambda d: (d['trend_15'] > 20) & (d['dmi_diff'] > 3), 'LONG'),
        ('trend < -20 + dmi < -3 → SHORT', lambda d: (d['trend_15'] < -20) & (d['dmi_diff'] < -3), 'SHORT'),

        # Exhaustion: extreme z + high volume
        ('z > 3 + vol_rel > 2 → SHORT', lambda d: (d['z_se'] > 3) & (d['vol_rel_live'] > 2), 'SHORT'),
        ('z < -3 + vol_rel > 2 → LONG', lambda d: (d['z_se'] < -3) & (d['vol_rel_live'] > 2), 'LONG'),

        # Trend + high volume (climax)
        ('trend > 20 + vol > 2 → SHORT (fade)', lambda d: (d['trend_15'] > 20) & (d['vol_rel_live'] > 2), 'SHORT'),
        ('trend < -20 + vol > 2 → LONG (fade)', lambda d: (d['trend_15'] < -20) & (d['vol_rel_live'] > 2), 'LONG'),

        # Low vol quiet market reversion
        ('z > 2 + vol < 0.5 → SHORT', lambda d: (d['z_se'] > 2) & (d['vol_rel_live'] < 0.5), 'SHORT'),
        ('z < -2 + vol < 0.5 → LONG', lambda d: (d['z_se'] < -2) & (d['vol_rel_live'] < 0.5), 'LONG'),

        # Nightmare + trend alignment (reversion WITH trend)
        ('z > 2 + trend < -10 → SHORT', lambda d: (d['z_se'] > 2) & (d['trend_15'] < -10), 'SHORT'),
        ('z < -2 + trend > 10 → LONG', lambda d: (d['z_se'] < -2) & (d['trend_15'] > 10), 'LONG'),

        # Nightmare + trend opposing (reversion AGAINST trend — risky)
        ('z > 2 + trend > 10 → SHORT', lambda d: (d['z_se'] > 2) & (d['trend_15'] > 10), 'SHORT'),
        ('z < -2 + trend < -10 → LONG', lambda d: (d['z_se'] < -2) & (d['trend_15'] < -10), 'LONG'),

        # DMI extreme + low variance (stable strong move)
        ('dmi > 5 + vr < 0.5 → LONG', lambda d: (d['dmi_diff'] > 5) & (d['variance_ratio'] < 0.5), 'LONG'),
        ('dmi < -5 + vr < 0.5 → SHORT', lambda d: (d['dmi_diff'] < -5) & (d['variance_ratio'] < 0.5), 'SHORT'),

        # Time-based: US session + extreme z
        ('z > 2 + US session → SHORT', lambda d: (d['z_se'] > 2) & (d['hour'] >= 13) & (d['hour'] <= 20), 'SHORT'),
        ('z < -2 + US session → LONG', lambda d: (d['z_se'] < -2) & (d['hour'] >= 13) & (d['hour'] <= 20), 'LONG'),

        # Asia session + low vol reversion
        ('z > 2 + Asia session → SHORT', lambda d: (d['z_se'] > 2) & (d['hour'] >= 0) & (d['hour'] <= 7), 'SHORT'),
        ('z < -2 + Asia session → LONG', lambda d: (d['z_se'] < -2) & (d['hour'] >= 0) & (d['hour'] <= 7), 'LONG'),

        # High bar range + z extreme (volatile reversion)
        ('z > 2 + range > 40 → SHORT', lambda d: (d['z_se'] > 2) & (d['bar_range'] > 40), 'SHORT'),
        ('z < -2 + range > 40 → LONG', lambda d: (d['z_se'] < -2) & (d['bar_range'] > 40), 'LONG'),

        # Triple: z + vr + vol
        ('z > 2 + vr < 0.5 + vol > 1.5 → SHORT', lambda d: (d['z_se'] > 2) & (d['variance_ratio'] < 0.5) & (d['vol_rel_live'] > 1.5), 'SHORT'),
        ('z < -2 + vr < 0.5 + vol > 1.5 → LONG', lambda d: (d['z_se'] < -2) & (d['variance_ratio'] < 0.5) & (d['vol_rel_live'] > 1.5), 'LONG'),

        # Triple: z + trend + dmi all aligned
        ('z > 2 + trend < -10 + dmi < -3 → SHORT', lambda d: (d['z_se'] > 2) & (d['trend_15'] < -10) & (d['dmi_diff'] < -3), 'SHORT'),
        ('z < -2 + trend > 10 + dmi > 3 → LONG', lambda d: (d['z_se'] < -2) & (d['trend_15'] > 10) & (d['dmi_diff'] > 3), 'LONG'),
    ]

    for label, mask_fn, direction in tqdm(combos, desc='  Combos'):
        try:
            mask = mask_fn(df)
        except Exception:
            continue

        pnl_prefix = 'fwd_{}_long' if direction == 'LONG' else 'fwd_{}_short'

        for horizon in HORIZONS:
            pnl_col = pnl_prefix.format(horizon)
            sub = df.loc[mask, pnl_col].dropna()
            if len(sub) < MIN_TRADES:
                continue

            wr = (sub > 0).mean() * 100
            avg = sub.mean()
            total = sub.sum()
            sharpe = sub.mean() / sub.std() if sub.std() > 0 else 0

            results.append({
                'strategy': label,
                'condition': label.rsplit(' → ', 1)[0],
                'direction': direction,
                'horizon': horizon,
                'n_trades': len(sub),
                'wr': wr,
                'avg_pnl': avg,
                'total_pnl': total,
                'sharpe': sharpe,
                'median_pnl': sub.median(),
            })

    return results


def generate_report(results_df, label):
    """Generate ranked strategy report."""
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)

    date_str = datetime.now().strftime('%Y%m%d')
    csv_path = os.path.join(out_dir, f'strategy_miner_{label}_{date_str}.csv')
    md_path = os.path.join(out_dir, f'strategy_miner_{label}_{date_str}.md')

    results_df.to_csv(csv_path, index=False)
    print(f'  Raw results: {csv_path} ({len(results_df)} entries)')

    # Filter: positive sharpe, WR > 50% or avg_pnl > $2
    good = results_df[
        (results_df['sharpe'] > 0.02) &
        (results_df['avg_pnl'] > 1.0) &
        (results_df['n_trades'] >= MIN_TRADES)
    ].copy()

    # Rank by sharpe * sqrt(n_trades) — rewards both edge quality and frequency
    good['score'] = good['sharpe'] * np.sqrt(good['n_trades'])
    good = good.sort_values('score', ascending=False)

    # Deduplicate: keep best horizon for each strategy
    best = good.groupby('condition').first().reset_index()
    best = best.sort_values('score', ascending=False).head(30)

    lines = []
    lines.append(f'# Strategy Miner Results — {label.upper()}')
    lines.append(f'')
    lines.append(f'**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'**Data**: {"OOS (2026)" if OOS else "IS (2025)"}')
    lines.append(f'**Min trades**: {MIN_TRADES}')
    lines.append(f'**Horizons scanned**: {HORIZONS}')
    lines.append(f'**Total scans**: {len(results_df)}')
    lines.append(f'**Profitable strategies found**: {len(good)}')
    lines.append(f'')

    lines.append(f'## Top 30 Strategies (ranked by Sharpe x sqrt(N))')
    lines.append(f'')
    lines.append(f'| # | Strategy | Dir | Hz | N | WR | Avg | Med | Total | Sharpe | Score |')
    lines.append(f'|---|----------|-----|-----|---|----|----|-----|-------|--------|-------|')

    for i, (_, row) in enumerate(best.iterrows()):
        lines.append(
            f'| {i+1} | {row["condition"]} | {row["direction"]} | {int(row["horizon"])} | '
            f'{int(row["n_trades"])} | {row["wr"]:.0f}% | ${row["avg_pnl"]:.1f} | '
            f'${row["median_pnl"]:.1f} | ${row["total_pnl"]:.0f} | {row["sharpe"]:.3f} | '
            f'{row["score"]:.2f} |'
        )

    lines.append(f'')
    lines.append(f'## Strategy Groups')
    lines.append(f'')

    # Group by type
    reversion = best[best['condition'].str.contains('z_se')]
    trend = best[best['condition'].str.contains('trend')]
    vol_based = best[best['condition'].str.contains('vol')]

    if len(reversion) > 0:
        lines.append(f'### Reversion Strategies (z_se based): {len(reversion)}')
        for _, row in reversion.head(5).iterrows():
            lines.append(f'  - {row["condition"]} → {row["direction"]} (H{int(row["horizon"])}): '
                        f'WR={row["wr"]:.0f}%, ${row["avg_pnl"]:.1f}/trade, N={int(row["n_trades"])}')
        lines.append('')

    if len(trend) > 0:
        lines.append(f'### Trend Strategies: {len(trend)}')
        for _, row in trend.head(5).iterrows():
            lines.append(f'  - {row["condition"]} → {row["direction"]} (H{int(row["horizon"])}): '
                        f'WR={row["wr"]:.0f}%, ${row["avg_pnl"]:.1f}/trade, N={int(row["n_trades"])}')
        lines.append('')

    if len(vol_based) > 0:
        lines.append(f'### Volume Strategies: {len(vol_based)}')
        for _, row in vol_based.head(5).iterrows():
            lines.append(f'  - {row["condition"]} → {row["direction"]} (H{int(row["horizon"])}): '
                        f'WR={row["wr"]:.0f}%, ${row["avg_pnl"]:.1f}/trade, N={int(row["n_trades"])}')
        lines.append('')

    lines.append(f'## Implementation Notes')
    lines.append(f'')
    lines.append(f'- These are raw forward-return scans, NOT zero-lookahead ticker results')
    lines.append(f'- Must validate with nightmare_ticker.py before trusting')
    lines.append(f'- Overlapping conditions exist — strategies may fire simultaneously')
    lines.append(f'- Sharpe computed per-trade, not time-series (no autocorrelation adjustment)')
    lines.append(f'- Score = Sharpe * sqrt(N) — balances edge quality with trade frequency')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Report: {md_path}')


def main():
    label = 'oos' if OOS else 'is'
    print(f'STRATEGY MINER — {label.upper()}')
    print(f'  Min trades: {MIN_TRADES}')
    print(f'  Horizons: {HORIZONS}')
    print()

    # Load and enrich data
    df = load_1m_with_features(oos=OOS)
    df = compute_forward_returns(df)

    # Handle the feature name mapping (SFE output names vs our names)
    if 'variance_ratio' not in df.columns and 'vr' in FEATURE_NAMES_13D:
        # Map from FEATURE_NAMES_13D to our expected names
        pass  # they should already be named by FEATURE_NAMES_13D

    # Ensure key columns exist
    for col in ['z_se', 'variance_ratio', 'dmi_diff', 'bar_range']:
        if col not in df.columns:
            print(f'  WARNING: {col} not in columns, checking alternatives...')
            # Try the 13D feature names
            for i, name in enumerate(FEATURE_NAMES_13D):
                if col in name or name in col:
                    print(f'    Found: {name}')

    print()

    # Phase 1: Single feature scans
    print('Phase 1: Single feature scans...')
    all_results = []

    for feat_name, feat_idx, thresholds, rule_type in SCAN_RULES:
        # Find the column name
        if feat_name in df.columns:
            col = feat_name
        elif feat_idx is not None and feat_idx < len(FEATURE_NAMES_13D):
            col = FEATURE_NAMES_13D[feat_idx]
        else:
            print(f'  Skipping {feat_name}: column not found')
            continue

        results = scan_single_feature(df, feat_name, col, thresholds, rule_type)
        all_results.extend(results)

    print(f'  {len(all_results)} single-feature results')

    # Phase 2: Combination scans
    print('Phase 2: Combination scans...')
    combo_results = scan_combinations(df)
    all_results.extend(combo_results)
    print(f'  {len(combo_results)} combo results')

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)
    print(f'\nTotal results: {len(results_df)}')

    # Generate report
    generate_report(results_df, label)

    print(f'\nDone.')


if __name__ == '__main__':
    main()

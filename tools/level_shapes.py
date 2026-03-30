"""
Level Shape Analysis: what does price action look like around levels?

Extracts windows of bars around every level touch and classifies:
1. REVERSAL: price touched level and turned back
2. BREAKOUT: price touched level and went through
3. PLATEAU: price sat at level (small bars, consolidation)
4. BOUNCE: price hit level and bounced hard (big bar rejection)

Outputs: averaged candlestick shapes per type, feature profiles, charts.

Usage:
  python -m tools.level_shapes
  python -m tools.level_shapes --months 2025-01
"""
import argparse
import gc
import glob
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.directory'] = os.path.abspath('examples')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS = 'DATA/ATLAS'
TICK = 0.25
OUT_DIR = 'reports/findings/shapes'
WINDOW = 10  # bars before and after the touch

FEATURE_NAMES = [
    'dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity', 'z_se', 'price_accel',
    'std_price', 'variance_ratio', 'bar_range', 'wick_ratio',
    'vwap_distance', 'time_of_day',
]


def load_month(tf, month_str):
    """Load OHLCV + 13D features for a TF/month."""
    from core.statistical_field_engine import StatisticalFieldEngine
    from training.train_trade_cnn import extract_features_13d

    files = sorted(glob.glob(os.path.join(ATLAS, tf, '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    m_start = pd.Timestamp(f'{month_str}-01').timestamp()
    m = int(month_str[5:7])
    y = int(month_str[:4])
    m_end = pd.Timestamp(f'{y}-{m+1:02d}-01').timestamp() if m < 12 else pd.Timestamp(f'{y+1}-01-01').timestamp()
    df = df[(df['timestamp'] >= m_start) & (df['timestamp'] < m_end)].reset_index(drop=True)

    if len(df) < 30:
        return None, None

    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    feats = extract_features_13d(states, df)
    del states; gc.collect()
    return df, feats


def get_levels(month_str):
    """Find levels for this month."""
    files = sorted(glob.glob('DATA/levels/levels_*.json'))
    best = None
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d['date'][:7] <= month_str:
            best = d['levels']
    return best


def classify_touch(closes, highs, lows, touch_idx, level_price, level_type):
    """Classify what happened when price touched a level.

    Returns: 'reversal', 'breakout', 'plateau', 'bounce', or None
    """
    n = len(closes)
    i = touch_idx

    if i < 3 or i >= n - 5:
        return None

    # Bar properties at touch
    bar_body = abs(closes[i] - closes[i-1])
    bar_range = highs[i] - lows[i]
    wick = bar_range - abs(closes[i] - (highs[i] + lows[i]) / 2)

    # What happened in next 5 bars?
    future_closes = closes[i+1:min(i+6, n)]
    if len(future_closes) < 3:
        return None

    # Direction before touch (last 3 bars)
    approach_dir = closes[i] - closes[max(0, i-3)]

    # Direction after touch (next 3 bars)
    depart_dir = future_closes[2] - closes[i] if len(future_closes) > 2 else 0

    # Movement magnitude
    future_range = max(future_closes) - min(future_closes)

    if level_type == 'resistance':
        # At resistance: approaching from below
        reversed_here = depart_dir < -bar_range * 0.3  # moved away down
        broke_through = future_closes[-1] > level_price + 20  # closed above
    else:
        # At support: approaching from above
        reversed_here = depart_dir > bar_range * 0.3  # moved away up
        broke_through = future_closes[-1] < level_price - 20  # closed below

    # Classify
    if broke_through:
        return 'breakout'
    elif reversed_here and bar_range > np.median(highs - lows) * 1.5:
        return 'bounce'  # big rejection bar
    elif reversed_here:
        return 'reversal'
    elif future_range < bar_range * 0.5:
        return 'plateau'  # sat there, didn't move
    else:
        return 'reversal'  # default to reversal if not breakout


def extract_shapes(df, feats, levels):
    """Extract windows around every level touch."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    n = len(closes)

    level_prices = [(l['price'], l['type']) for l in levels]
    shapes = {'reversal': [], 'breakout': [], 'plateau': [], 'bounce': []}
    feature_profiles = {'reversal': [], 'breakout': [], 'plateau': [], 'bounce': []}

    for lp, ltype in level_prices:
        for i in range(WINDOW, n - WINDOW):
            # Did this bar touch the level? (wick reached within 15 points)
            touched = (lows[i] <= lp + 15 and highs[i] >= lp - 15)
            if not touched:
                continue

            classification = classify_touch(closes, highs, lows, i, lp, ltype)
            if classification is None:
                continue

            # Extract price shape: normalize to level price
            window_slice = slice(i - WINDOW, i + WINDOW + 1)
            shape = {
                'close': (closes[window_slice] - lp) / TICK,
                'high': (highs[window_slice] - lp) / TICK,
                'low': (lows[window_slice] - lp) / TICK,
                'open': (opens[window_slice] - lp) / TICK,
            }

            if len(shape['close']) == 2 * WINDOW + 1:
                shapes[classification].append(shape)

            # Extract feature profile
            feat_window = feats[window_slice]
            if len(feat_window) == 2 * WINDOW + 1:
                feature_profiles[classification].append(feat_window)

    return shapes, feature_profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='1h', choices=['1D', '4h', '1h', '15m', '5m', '1m', '15s', '5s', '1s'],
                        help='Timeframe for shape analysis')
    parser.add_argument('--months', default=None, help='Comma-separated YYYY-MM')
    args = parser.parse_args()

    OUT_DIR = f'reports/findings/shapes/{args.tf}'
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.months:
        months = args.months.split(',')
    else:
        level_files = sorted(glob.glob('DATA/levels/levels_*.json'))
        months = sorted(set(json.load(open(f))['date'][:7] for f in level_files))

    print(f"Level Shape Analysis — {len(months)} months")

    all_shapes = {'reversal': [], 'breakout': [], 'plateau': [], 'bounce': []}
    all_feat_profiles = {'reversal': [], 'breakout': [], 'plateau': [], 'bounce': []}

    for month in tqdm(months, desc="Months"):
        levels = get_levels(month)
        if not levels or len(levels) < 2:
            continue

        # For large TFs (1s, 15s): process per-month parquet directly to avoid OOM
        tf_folder = os.path.join(ATLAS, args.tf)
        month_file = os.path.join(tf_folder, f'{month.replace("-", "_")}.parquet')

        if args.tf in ('1s', '15s', '5s') and os.path.exists(month_file):
            # Sharded: load just this month's parquet
            from core.statistical_field_engine import StatisticalFieldEngine
            from training.train_trade_cnn import extract_features_13d

            df_month = pd.read_parquet(month_file).sort_values('timestamp').reset_index(drop=True)
            if len(df_month) < 30:
                continue
            tqdm.write(f"  {args.tf} {month}: {len(df_month):,} bars (sharded)")

            sfe = StatisticalFieldEngine()
            states = sfe.batch_compute_states(df_month)
            feats = extract_features_13d(states, df_month)
            del states
            del sfe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            shapes, feat_profiles = extract_shapes(df_month, feats, levels)
            for cls in shapes:
                all_shapes[cls].extend(shapes[cls])
                all_feat_profiles[cls].extend(feat_profiles[cls])

            del df_month, feats, shapes, feat_profiles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            df, feats = load_month(args.tf, month)
            if df is None:
                continue

            shapes, feat_profiles = extract_shapes(df, feats, levels)
            for cls in shapes:
                all_shapes[cls].extend(shapes[cls])
                all_feat_profiles[cls].extend(feat_profiles[cls])

            del df, feats, shapes, feat_profiles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"SHAPE SUMMARY")
    print(f"{'='*60}")
    for cls in ['reversal', 'breakout', 'plateau', 'bounce']:
        print(f"  {cls:>10}: {len(all_shapes[cls])} events")

    # Chart 1: Average price shape per classification
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Average Price Shape at Levels (normalized to level=0)', fontsize=14, fontweight='bold')

    x = np.arange(-WINDOW, WINDOW + 1)
    colors = {'reversal': '#2ECC71', 'breakout': '#E74C3C', 'plateau': '#F39C12', 'bounce': '#3498DB'}

    for idx, cls in enumerate(['reversal', 'breakout', 'plateau', 'bounce']):
        ax = axes[idx // 2][idx % 2]
        if not all_shapes[cls]:
            ax.set_title(f'{cls.upper()} (no data)')
            continue

        close_avg = np.mean([s['close'] for s in all_shapes[cls]], axis=0)
        high_avg = np.mean([s['high'] for s in all_shapes[cls]], axis=0)
        low_avg = np.mean([s['low'] for s in all_shapes[cls]], axis=0)

        ax.fill_between(x, low_avg, high_avg, alpha=0.2, color=colors[cls])
        ax.plot(x, close_avg, color=colors[cls], linewidth=2, label='Avg close')
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5, label='Level')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_title(f'{cls.upper()} ({len(all_shapes[cls])} events)', fontsize=12)
        ax.set_xlabel('Bars from touch')
        ax.set_ylabel('Ticks from level')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'price_shapes.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart: {OUT_DIR}/price_shapes.png")

    # Chart 2: Feature profiles per classification (top 6 features from EDA)
    key_features = ['z_se', 'dmi_diff', 'velocity', 'vol_rel', 'bar_range', 'wick_ratio']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Feature Profiles Around Level Touches', fontsize=14, fontweight='bold')

    for fi, fname in enumerate(key_features):
        ax = axes[fi // 3][fi % 3]
        feat_idx = FEATURE_NAMES.index(fname)

        for cls in ['reversal', 'breakout', 'bounce']:
            if not all_feat_profiles[cls]:
                continue
            profiles = np.array([fp[:, feat_idx] for fp in all_feat_profiles[cls]])
            avg = profiles.mean(axis=0)
            ax.plot(x, avg, color=colors[cls], linewidth=2, label=cls)

        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_title(fname, fontsize=11)
        ax.set_xlabel('Bars from touch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'feature_profiles.png'), dpi=150, bbox_inches='tight')
    print(f"  Chart: {OUT_DIR}/feature_profiles.png")

    # Save raw counts
    summary = {cls: len(all_shapes[cls]) for cls in all_shapes}
    with open(os.path.join(OUT_DIR, 'shape_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {OUT_DIR}/shape_summary.json")


if __name__ == '__main__':
    main()

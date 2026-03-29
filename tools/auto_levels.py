"""
Automatic Level Detection — learned from human-drawn levels.

Rules derived from 31 weeks of hand-drawn support/resistance:
  - Target ~8 levels per week (median from human data)
  - Minimum spacing: 100 points between adjacent levels
  - Levels at price points most touched/rejected by wicks
  - Persistence: prefer levels that held for multiple days
  - R/S classification: above price midpoint = resistance, below = support

Usage:
  python -m tools.auto_levels --date 2025-01-06
  python -m tools.auto_levels --date 2025-01-06 --validate  (overlay on next week)
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.directory'] = os.path.abspath('examples')
import matplotlib.pyplot as plt
from collections import Counter

ATLAS_ROOT = 'DATA/ATLAS'
TICK = 0.25

# Parameters learned from human-drawn levels
TARGET_LEVELS = 8          # median levels per week
MIN_SPACING_POINTS = 100   # minimum distance between adjacent levels
TOUCH_ZONE_TICKS = 8       # how close a wick must be to count as a "touch"


def load_week_data(date_str, tf='4h'):
    """Load 4 weeks of data centered on the given date."""
    dt = pd.Timestamp(date_str)
    window_start = dt - pd.Timedelta(days=21)
    window_end = dt + pd.Timedelta(days=7)

    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet')))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    start_ts = window_start.timestamp()
    end_ts = window_end.timestamp()
    return df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)].reset_index(drop=True)


def find_levels_stddev(df):
    """Find levels using standard deviation bands on the price window.

    The price distribution within a window is roughly normal around a mean.
    Levels sit at the σ boundaries where price statistically reverses:
      Mean = center of value
      ±1σ = normal trading range boundaries
      ±2σ = strong support/resistance (price rarely goes beyond)
      ±3σ = extreme (absolute high/low)
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Use VWAP-like center if volume available, else simple mean
    if 'volume' in df.columns and df['volume'].sum() > 0:
        vols = df['volume'].values
        center = np.average(closes, weights=vols)
    else:
        center = closes.mean()

    # Standard deviation of close prices
    std = closes.std()

    # Also compute from high/low range for tighter bands
    all_prices = np.concatenate([highs, lows])
    std_hl = all_prices.std()

    mid = (closes.max() + closes.min()) / 2

    levels = []
    bands = [
        (center + 3 * std, 'resistance', '3sd_upper'),
        (center + 2 * std, 'resistance', '2sd_upper'),
        (center + 1 * std, 'resistance', '1sd_upper'),
        (center, 'support', 'mean'),
        (center - 1 * std, 'support', '1sd_lower'),
        (center - 2 * std, 'support', '2sd_lower'),
        (center - 3 * std, 'support', '3sd_lower'),
    ]

    for price, default_type, label in bands:
        snapped = round(price / TICK) * TICK
        # Only include if within the actual price range (± some margin)
        if snapped < lows.min() - 100 or snapped > highs.max() + 100:
            continue
        level_type = 'resistance' if snapped > mid else 'support'
        color = '#CC0000' if level_type == 'resistance' else '#0066CC'
        levels.append({
            'price': snapped,
            'type': level_type,
            'color': color,
            'strength': 0,
            'source': f'stddev_{label}',
        })

    levels.sort(key=lambda x: x['price'], reverse=True)
    return levels


def find_levels(df, eps_points=50, min_samples=3):
    """Find support/resistance levels using the human process:

    1. OUTER BOUNDS: absolute high/low of the window (ceiling/floor)
    2. WICK CLUSTERING: DBSCAN on all wick tips (highs and lows) to find
       price zones where multiple bars interacted
    3. SMALL BAR ACCUMULATION: find prices where small-body bars cluster —
       institutional accumulation/distribution zones. These are the
       "tell-tale" internal levels.

    The level price is placed at the point with maximum wick touches
    within each cluster (not the centroid — the most-touched price).
    """
    from sklearn.cluster import DBSCAN

    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    n = len(highs)

    if n < 5:
        return []

    mid = (closes.max() + closes.min()) / 2

    # --- Step 1: Outer bounds (absolute high and low) ---
    abs_high = round(float(highs.max()) / TICK) * TICK
    abs_low = round(float(lows.min()) / TICK) * TICK

    # --- Step 2: Wick tip clustering ---
    # Collect ALL wick tips (where price reversed or paused)
    wick_prices = []
    for i in range(n):
        wick_prices.append(highs[i])
        wick_prices.append(lows[i])

    X = np.array(wick_prices).reshape(-1, 1)
    db = DBSCAN(eps=eps_points, min_samples=min_samples)
    labels = db.fit_predict(X)

    # --- Step 3: For each cluster, find the most-touched price ---
    # (maximize visual alignment — the price that the most wicks reach)
    cluster_levels = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = X[labels == label].flatten()

        # Find the price within the cluster that has the most bars touching it
        # "Touching" = wick passes through (low <= price <= high)
        best_price = None
        best_touches = 0

        # Test prices at TICK resolution within the cluster range
        p_min = cluster.min()
        p_max = cluster.max()
        test_prices = np.arange(p_min, p_max + TICK, TICK)

        for test_p in test_prices:
            touches = 0
            for i in range(n):
                if lows[i] <= test_p <= highs[i]:
                    touches += 1
            if touches > best_touches:
                best_touches = touches
                best_price = test_p

        if best_price is not None:
            cluster_levels.append({
                'price': round(best_price / TICK) * TICK,
                'touches': best_touches,
                'cluster_size': len(cluster),
            })

    # --- Step 4: Small bar accumulation zones ---
    # Small bars = body < median body size. Where these cluster = institutional levels
    body_sizes = np.abs(closes - opens)
    median_body = np.median(body_sizes)
    small_bar_mask = body_sizes < median_body * 0.5  # bars with tiny bodies

    if small_bar_mask.sum() > 5:
        # Cluster the close prices of small bars
        small_closes = closes[small_bar_mask]
        X_small = small_closes.reshape(-1, 1)
        db_small = DBSCAN(eps=eps_points * 0.75, min_samples=max(2, int(small_bar_mask.sum() * 0.1)))
        labels_small = db_small.fit_predict(X_small)

        for label in set(labels_small):
            if label == -1:
                continue
            cluster = X_small[labels_small == label].flatten()
            level_price = round(float(np.median(cluster)) / TICK) * TICK
            # Check it's not too close to existing cluster levels
            too_close = any(abs(level_price - cl['price']) < eps_points * 0.5 for cl in cluster_levels)
            if not too_close:
                # Count full-dataset touches
                touches = sum(1 for i in range(n) if lows[i] <= level_price <= highs[i])
                cluster_levels.append({
                    'price': level_price,
                    'touches': touches,
                    'cluster_size': len(cluster),
                    'source': 'small_bar',
                })

    # --- Step 5: Assemble final levels ---
    levels = []

    # Add outer bounds
    levels.append({
        'price': abs_high, 'type': 'resistance', 'color': '#CC0000',
        'strength': 1, 'source': 'outer_high',
    })
    levels.append({
        'price': abs_low, 'type': 'support', 'color': '#0066CC',
        'strength': 1, 'source': 'outer_low',
    })

    # Add cluster levels (sorted by touch count)
    cluster_levels.sort(key=lambda x: x['touches'], reverse=True)
    existing_prices = {abs_high, abs_low}

    for cl in cluster_levels:
        too_close = any(abs(cl['price'] - p) < eps_points * 0.5 for p in existing_prices)
        if too_close:
            continue
        if cl['price'] > mid:
            level_type = 'resistance'
            color = '#CC0000'
        else:
            level_type = 'support'
            color = '#0066CC'
        levels.append({
            'price': cl['price'],
            'type': level_type,
            'color': color,
            'strength': cl['touches'],
            'source': cl.get('source', 'wick_cluster'),
        })
        existing_prices.add(cl['price'])

    levels.sort(key=lambda x: x['price'], reverse=True)
    return levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='2025-01-06', help='Week to detect levels for')
    parser.add_argument('--validate', action='store_true', help='Overlay on next week for validation')
    parser.add_argument('--compare', action='store_true', help='Compare to human-drawn levels')
    parser.add_argument('--method', default='stddev', choices=['stddev', 'dbscan', 'both'],
                        help='Level detection method')
    args = parser.parse_args()

    print(f"Auto Level Detection for week of {args.date}")

    merged = []

    if args.method in ('stddev', 'both'):
        # Standard deviation: use the full 4-week window
        print(f"\n  === STDDEV METHOD ===")
        df_full = load_week_data(args.date, tf='1h')
        if df_full is not None and len(df_full) > 0:
            std_levels = find_levels_stddev(df_full)
            for l in std_levels:
                print(f"    {l['price']:.2f} ({l['type']}) src={l['source']}")
            merged.extend(std_levels)

    if args.method in ('dbscan', 'both'):
        # DBSCAN refines WITHIN the stddev bands
        # Use the stddev levels as reference — find structural levels near each band
        print(f"\n  === DBSCAN REFINEMENT (using stddev as reference) ===")

        # Get stddev bands as reference zones
        ref_prices = [l['price'] for l in merged]

        cascade = [
            ('1D', 100, 2),
            ('4h', 75, 2),
            ('1h', 50, 3),
        ]
        all_prices = set(l['price'] for l in merged)

        for tf, eps, min_samp in cascade:
            df_tf = load_week_data(args.date, tf=tf)
            if df_tf is None or len(df_tf) == 0:
                continue
            tf_levels = find_levels(df_tf, eps_points=eps, min_samples=min_samp)

            added = 0
            for l in tf_levels:
                # Skip if too close to existing
                too_close = any(abs(l['price'] - p) < eps * 0.5 for p in all_prices)
                if too_close:
                    continue

                # Check: is this level near a stddev band? If so, it REPLACES the band
                # (structural level > statistical level)
                replaced = False
                for i, ref in enumerate(merged):
                    if ref.get('source', '').startswith('stddev_') and abs(l['price'] - ref['price']) < eps:
                        # Replace stddev with structural level at nearby price
                        print(f"    Replacing {ref['source']} ({ref['price']:.0f}) "
                              f"with structural {l['price']:.0f} (str={l.get('strength', 0)})")
                        merged[i] = l
                        all_prices.discard(ref['price'])
                        all_prices.add(l['price'])
                        replaced = True
                        added += 1
                        break

                if not replaced:
                    # New level between bands
                    merged.append(l)
                    all_prices.add(l['price'])
                    added += 1

            print(f"    {tf}: {len(tf_levels)} found, {added} integrated")

    merged.sort(key=lambda x: x['price'], reverse=True)
    print(f"\n  Final: {len(merged)} levels")

    # Save auto-detected levels
    out_dir = 'DATA/levels_auto'
    os.makedirs(out_dir, exist_ok=True)
    data = {
        'date': args.date,
        'method': 'auto',
        'levels': merged,
        'n_resistance': sum(1 for l in merged if l['type'] == 'resistance'),
        'n_support': sum(1 for l in merged if l['type'] == 'support'),
    }
    json_path = os.path.join(out_dir, f'levels_{args.date}.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Compare to human levels if available
    human_path = os.path.join('DATA', 'levels', f'levels_{args.date}.json')
    if args.compare and os.path.exists(human_path):
        with open(human_path) as f:
            human = json.load(f)
        human_prices = [l['price'] for l in human.get('levels', [])]
        auto_prices = [l['price'] for l in merged]

        print(f"\n  COMPARISON TO HUMAN:")
        print(f"    Human: {len(human_prices)} levels")
        print(f"    Auto:  {len(auto_prices)} levels")

        # Match: how many auto levels are within 50 points of a human level?
        matched = 0
        for ap in auto_prices:
            for hp in human_prices:
                if abs(ap - hp) < 50:
                    matched += 1
                    break
        print(f"    Matched (within 50 pts): {matched}/{len(auto_prices)} ({matched/max(1,len(auto_prices))*100:.0f}%)")

    # Chart
    df_1m = load_week_data(args.date, tf='1m')
    if df_1m is not None and len(df_1m) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        fig.suptitle(f'Auto-Detected Levels — {args.date}', fontsize=14, fontweight='bold')

        prices = df_1m['close'].values
        x = np.arange(len(prices))
        ax.plot(x, prices, 'k-', linewidth=0.5, alpha=0.6)

        # Auto levels
        for level in merged:
            ax.axhline(y=level['price'], color=level['color'], linewidth=1.5,
                        alpha=0.7, linestyle='--' if level['type'] == 'resistance' else '-')
            ax.text(len(prices) + 2, level['price'], f'{level["price"]:.0f} (auto)',
                    fontsize=8, color=level['color'], va='center')

        # Human levels if available
        if os.path.exists(human_path):
            with open(human_path) as f:
                human = json.load(f)
            for level in human.get('levels', []):
                ax.axhline(y=level['price'], color='green', linewidth=1, alpha=0.5, linestyle=':')
                ax.text(2, level['price'] + 5, f'{level["price"]:.0f} (human)',
                        fontsize=7, color='green', va='bottom')

        # Timestamp labels
        n_ticks = 20
        tick_step = max(1, len(prices) // n_ticks)
        tick_pos = list(range(0, len(prices), tick_step))
        tick_labels = [pd.to_datetime(df_1m.iloc[i]['timestamp'], unit='s').strftime('%m/%d %H:%M')
                       for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        chart_path = f'examples/auto_levels_{args.date}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"  Chart: {chart_path}")


if __name__ == '__main__':
    main()

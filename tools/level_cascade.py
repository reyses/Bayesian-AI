"""
Level Cascade: Top-down horizontal line validation.

Start at 4h — draw support/resistance at highs/lows.
Each lower TF validates or adjusts the lines from above.
Bottom (1s) only trades within confirmed boundaries.

Usage:
  python -m tools.level_cascade --trade-date 2026-03-18 --lookback-start 2026-03-16
"""
import argparse
import gc
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICK = 0.25
ATLAS_ROOT = 'DATA/ATLAS'


def load_tf_day(tf, date_str):
    """Load bars for a specific date from ATLAS."""
    month = date_str[:7].replace('-', '_')
    path = os.path.join(ATLAS_ROOT, tf, f'{month}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    target = pd.Timestamp(date_str).date()
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    return df[df['date'] == target].reset_index(drop=True)


def load_tf_range(tf, start_date, end_date):
    """Load bars for a date range from ATLAS."""
    dfs = []
    for f in sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet'))):
        df = pd.read_parquet(f)
        dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    start_ts = pd.Timestamp(start_date).timestamp()
    end_ts = pd.Timestamp(end_date).timestamp() + 86400  # include end date
    return df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)].reset_index(drop=True)


def find_levels(prices, highs, lows, order=2):
    """Find horizontal support/resistance levels from local peaks and troughs."""
    if len(prices) < 5:
        return [], []

    peak_idx = argrelextrema(highs, np.greater, order=order)[0]
    trough_idx = argrelextrema(lows, np.less, order=order)[0]

    resistance = highs[peak_idx] if len(peak_idx) > 0 else np.array([])
    support = lows[trough_idx] if len(trough_idx) > 0 else np.array([])

    return resistance, support


def validate_levels(upper_resistance, upper_support, current_highs, current_lows, tolerance_ticks=4):
    """Validate upper TF levels against current TF price action.

    Returns: confirmed levels, broken levels, adjusted levels.
    """
    tol = tolerance_ticks * TICK
    confirmed_r = []
    broken_r = []
    confirmed_s = []
    broken_s = []

    for level in upper_resistance:
        # Check if current TF respected this level (price approached but didn't close above)
        approaches = np.sum(current_highs > level - tol)
        breaks = np.sum(current_highs > level + tol)
        if breaks > approaches * 0.3:  # more than 30% of approaches broke through
            broken_r.append(level)
        else:
            confirmed_r.append(level)

    for level in upper_support:
        approaches = np.sum(current_lows < level + tol)
        breaks = np.sum(current_lows < level - tol)
        if breaks > approaches * 0.3:
            broken_s.append(level)
        else:
            confirmed_s.append(level)

    return confirmed_r, broken_r, confirmed_s, broken_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trade-date', default='2026-03-18', help='Wednesday to trade')
    parser.add_argument('--lookback-start', default='2026-03-16', help='Monday for lookback')
    args = parser.parse_args()

    trade_date = args.trade_date
    lookback_start = args.lookback_start
    # Lookback ends at trade date start
    lookback_end = trade_date

    print(f"=" * 70)
    print(f"LEVEL CASCADE: Trade {trade_date} | Lookback {lookback_start} to {lookback_end}")
    print(f"=" * 70)

    # TF cascade: top to bottom
    tfs = ['4h', '1h', '15m', '5m', '1m', '15s', '5s', '1s']
    tf_order = {'4h': 2, '1h': 2, '15m': 3, '5m': 3, '1m': 4, '15s': 4, '5s': 5, '1s': 5}

    # Step 1: Load lookback data at 4h to establish initial levels
    print(f"\n--- STEP 1: 4h levels from {lookback_start} to {lookback_end} ---")
    df_4h = load_tf_range('4h', lookback_start, lookback_end)
    if df_4h is None or len(df_4h) == 0:
        # Fall back to 1h if no 4h data
        print("  No 4h data, using 1h as top level")
        df_4h = load_tf_range('1h', lookback_start, lookback_end)

    if df_4h is not None and len(df_4h) > 0:
        resistance_4h, support_4h = find_levels(
            df_4h['close'].values, df_4h['high'].values, df_4h['low'].values, order=2)
        # Also add the absolute high and low of the lookback
        abs_high = df_4h['high'].max()
        abs_low = df_4h['low'].min()
        resistance_4h = np.unique(np.append(resistance_4h, abs_high))
        support_4h = np.unique(np.append(support_4h, abs_low))
        print(f"  Bars: {len(df_4h)}")
        print(f"  Resistance: {[f'{r:.2f}' for r in sorted(resistance_4h)]}")
        print(f"  Support: {[f'{s:.2f}' for s in sorted(support_4h)]}")
    else:
        print("  No lookback data available")
        return

    # Step 2: Cascade down — each TF validates/refines levels from above
    current_resistance = resistance_4h
    current_support = support_4h
    tf_levels = {'4h': {'resistance': resistance_4h, 'support': support_4h}}

    cascade_tfs = ['1h', '15m', '5m', '1m']
    for tf in cascade_tfs:
        print(f"\n--- {tf.upper()}: validating against upper levels ---")
        df_tf = load_tf_range(tf, lookback_start, lookback_end)
        if df_tf is None or len(df_tf) == 0:
            print(f"  No data")
            tf_levels[tf] = {'resistance': current_resistance, 'support': current_support}
            continue

        highs = df_tf['high'].values
        lows = df_tf['low'].values

        # Validate upper levels
        conf_r, broken_r, conf_s, broken_s = validate_levels(
            current_resistance, current_support, highs, lows)

        # Find new levels at this TF
        new_r, new_s = find_levels(df_tf['close'].values, highs, lows, order=tf_order.get(tf, 3))

        # Merge: confirmed upper + new from this TF
        merged_r = np.unique(np.concatenate([conf_r, new_r])) if len(new_r) > 0 else np.array(conf_r)
        merged_s = np.unique(np.concatenate([conf_s, new_s])) if len(new_s) > 0 else np.array(conf_s)

        print(f"  Bars: {len(df_tf)}")
        print(f"  Upper levels confirmed: R={len(conf_r)} S={len(conf_s)}")
        print(f"  Upper levels broken:    R={len(broken_r)} S={len(broken_s)}")
        print(f"  New levels found:       R={len(new_r)} S={len(new_s)}")
        print(f"  Merged:                 R={len(merged_r)} S={len(merged_s)}")

        current_resistance = merged_r
        current_support = merged_s
        tf_levels[tf] = {'resistance': merged_r, 'support': merged_s}

    # Step 3: Load trade day data at 1m for the chart
    print(f"\n--- TRADE DAY: {trade_date} ---")
    df_trade = load_tf_day('1m', trade_date)
    if df_trade is None or len(df_trade) == 0:
        print("  No 1m data for trade date")
        return
    print(f"  1m bars: {len(df_trade)}")

    # Final levels going into trade day
    print(f"\n  FINAL LEVELS (entering {trade_date}):")
    print(f"  Resistance ({len(current_resistance)}):")
    for r in sorted(current_resistance, reverse=True):
        print(f"    {r:.2f}")
    print(f"  Support ({len(current_support)}):")
    for s in sorted(current_support, reverse=True):
        print(f"    {s:.2f}")

    # Step 4: Chart — trade day price with all cascade levels
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    fig.suptitle(f'Level Cascade — {trade_date} | Lookback {lookback_start}-{lookback_end}',
                 fontsize=14, fontweight='bold')

    # Plot trade day 1m price
    prices = df_trade['close'].values
    t_min = df_trade['timestamp'].min()
    x = (df_trade['timestamp'].values - t_min) / 60.0
    ax.plot(x, prices, 'k-', linewidth=1, alpha=0.8, label='1m price')

    # Plot levels from each TF in different colors
    level_colors = {'4h': '#CC0000', '1h': '#FF4444', '15m': '#FF8C00', '5m': '#9370DB', '1m': '#0066CC'}
    level_styles = {'4h': '-', '1h': '--', '15m': '-.', '5m': ':', '1m': ':'}

    for tf in ['4h', '1h', '15m', '5m', '1m']:
        if tf not in tf_levels:
            continue
        levels = tf_levels[tf]
        color = level_colors.get(tf, 'gray')
        ls = level_styles.get(tf, '--')

        for r in levels['resistance']:
            ax.axhline(y=r, color=color, linestyle=ls, linewidth=1.5, alpha=0.6)
        for s in levels['support']:
            ax.axhline(y=s, color=color, linestyle=ls, linewidth=1.5, alpha=0.6)

    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', linewidth=1, label='1m price')]
    for tf in ['4h', '1h', '15m', '5m', '1m']:
        if tf in tf_levels:
            n_r = len(tf_levels[tf]['resistance'])
            n_s = len(tf_levels[tf]['support'])
            legend_elements.append(
                Line2D([0], [0], color=level_colors.get(tf, 'gray'),
                       linestyle=level_styles.get(tf, '--'), linewidth=1.5,
                       label=f'{tf} levels (R={n_r} S={n_s})'))
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

    ax.set_xlabel('Minutes from session start')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    chart_path = f'reports/findings/level_cascade_{trade_date}.png'
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")

    # Save report
    report_path = f'reports/findings/level_cascade_{trade_date}.txt'
    with open(report_path, 'w') as f:
        f.write(f"Level Cascade Report — {trade_date}\n")
        f.write(f"Lookback: {lookback_start} to {lookback_end}\n\n")
        for tf in ['4h', '1h', '15m', '5m', '1m']:
            if tf not in tf_levels:
                continue
            f.write(f"\n{tf.upper()} LEVELS:\n")
            f.write(f"  Resistance: {[f'{r:.2f}' for r in sorted(tf_levels[tf]['resistance'], reverse=True)]}\n")
            f.write(f"  Support:    {[f'{s:.2f}' for s in sorted(tf_levels[tf]['support'])]}\n")
    print(f"  Report saved: {report_path}")


if __name__ == '__main__':
    main()

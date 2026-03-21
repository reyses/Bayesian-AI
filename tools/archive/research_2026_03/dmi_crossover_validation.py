#!/usr/bin/env python
"""
DMI Crossover Validation — Can DMI+/DMI- crossovers identify $10+ regime moves?

For each DI+/DI- crossover on 1m data:
  1. Direction: DI+ crosses above DI- = LONG, below = SHORT
  2. Load 1s data for next 10 minutes
  3. Measure MFE, MAE (before MFE), time to MFE
  4. Apply $10 SL (20 ticks): does the trade survive?
  5. Report: win rate, avg reward, false signal rate

Usage:
    python tools/dmi_crossover_validation.py                    # full IS dataset
    python tools/dmi_crossover_validation.py --months 2025_07   # single month
    python tools/dmi_crossover_validation.py --lookahead 600    # 10-min window (default)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.physics_utils import compute_adx_dmi_cpu
from tools.golden_path import load_1s_index, load_1s_window


# ── Constants ──────────────────────────────────────────────────────────────────
TICK_SIZE = 0.25
POINT_VALUE = 2.0  # MNQ: $2 per point, $0.50 per tick
SL_TICKS = 20      # $10 risk = 20 ticks = 5 points
LOOKAHEAD_SECS = 600  # 10 minutes of 1s data after crossover
MIN_ADX = 0.0       # minimum ADX at crossover (0 = no filter)


def compute_dmi_from_ohlc(df, period=14):
    """Compute DMI+/DMI- from OHLC data (CPU, no CUDA needed).

    Returns DataFrame with columns: timestamp, close, dmi_plus, dmi_minus, adx,
    di_plus_prev, di_minus_prev
    """
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n = len(df)

    # Compute raw TR, +DM, -DM
    tr_raw = np.zeros(n)
    plus_dm_raw = np.zeros(n)
    minus_dm_raw = np.zeros(n)

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_raw[i] = max(hl, hc, lc)

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm_raw[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm_raw[i] = down_move

    # Wilder smoothing
    adx, dmi_plus, dmi_minus = compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, period)

    result = pd.DataFrame({
        'timestamp': df['timestamp'].values,
        'close': closes,
        'high': highs,
        'low': lows,
        'dmi_plus': dmi_plus,
        'dmi_minus': dmi_minus,
        'adx': adx,
    })

    # Add previous bar's DI values for crossover detection
    result['di_plus_prev'] = result['dmi_plus'].shift(1)
    result['di_minus_prev'] = result['dmi_minus'].shift(1)

    return result


def detect_crossovers(df, min_adx=0.0):
    """Find all DI+/DI- crossover bars.

    Bullish crossover: prev DI+ < prev DI- AND curr DI+ > curr DI-
    Bearish crossover: prev DI+ > prev DI- AND curr DI+ < curr DI-

    Returns list of dicts: {bar_idx, timestamp, direction, dmi_plus, dmi_minus, adx, close}
    """
    crossovers = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        if pd.isna(row['di_plus_prev']) or pd.isna(row['di_minus_prev']):
            continue

        prev_diff = row['di_plus_prev'] - row['di_minus_prev']
        curr_diff = row['dmi_plus'] - row['dmi_minus']

        # Crossover: sign change
        if prev_diff <= 0 and curr_diff > 0:
            direction = 'LONG'
        elif prev_diff >= 0 and curr_diff < 0:
            direction = 'SHORT'
        else:
            continue

        # Optional ADX filter
        if row['adx'] < min_adx:
            continue

        crossovers.append({
            'bar_idx': i,
            'timestamp': float(row['timestamp']),
            'direction': direction,
            'dmi_plus': float(row['dmi_plus']),
            'dmi_minus': float(row['dmi_minus']),
            'adx': float(row['adx']),
            'close': float(row['close']),
            'dmi_spread': abs(curr_diff),
        })

    return crossovers


def measure_crossover_outcome(crossover, index_1s, lookahead_secs, sl_ticks, cache):
    """For a single crossover, measure MFE/MAE from 1s data.

    Returns dict with outcome metrics, or None if no 1s data available.
    """
    ts_entry = crossover['timestamp']
    ts_end = ts_entry + lookahead_secs
    entry_price = crossover['close']
    direction = crossover['direction']

    # Load 1s window
    window = load_1s_window(index_1s, ts_entry, ts_end, cache)
    if len(window) < 5:
        return None

    prices = window['close'].values.astype(float)
    timestamps = window['timestamp'].values.astype(float)

    # Compute excursions in the crossover direction
    if direction == 'LONG':
        # Favorable = price goes up, Adverse = price goes down
        fav_excursions = (prices - entry_price) / TICK_SIZE
        adv_excursions = (entry_price - prices) / TICK_SIZE
    else:
        # SHORT: Favorable = price goes down, Adverse = price goes up
        fav_excursions = (entry_price - prices) / TICK_SIZE
        adv_excursions = (prices - entry_price) / TICK_SIZE

    # MFE = max favorable excursion
    mfe_idx = int(np.argmax(fav_excursions))
    mfe_ticks = float(fav_excursions[mfe_idx])
    mfe_time = float(timestamps[mfe_idx] - ts_entry)

    # MAE = max adverse excursion BEFORE MFE
    if mfe_idx > 0:
        mae_before_mfe = float(np.max(adv_excursions[:mfe_idx + 1]))
    else:
        mae_before_mfe = 0.0

    # Overall MAE (entire window)
    mae_total = float(np.max(adv_excursions))

    # SL hit? Check if adverse excursion exceeds SL before MFE
    sl_hit = mae_before_mfe >= sl_ticks

    # If SL hit, find where it was hit
    sl_hit_time = None
    if sl_hit:
        for j in range(len(adv_excursions)):
            if adv_excursions[j] >= sl_ticks:
                sl_hit_time = float(timestamps[j] - ts_entry)
                break

    # Dollar values
    mfe_dollars = mfe_ticks * TICK_SIZE * POINT_VALUE
    mae_dollars = mae_before_mfe * TICK_SIZE * POINT_VALUE
    sl_dollars = sl_ticks * TICK_SIZE * POINT_VALUE

    # Net PnL with $10 SL: if stopped out, lose $10; otherwise, gain MFE
    if sl_hit:
        net_pnl = -sl_dollars
    else:
        net_pnl = mfe_dollars

    return {
        'timestamp': ts_entry,
        'direction': direction,
        'entry_price': entry_price,
        'mfe_ticks': mfe_ticks,
        'mfe_dollars': mfe_dollars,
        'mae_before_mfe_ticks': mae_before_mfe,
        'mae_before_mfe_dollars': mae_dollars,
        'mae_total_ticks': mae_total,
        'time_to_mfe_secs': mfe_time,
        'sl_hit': sl_hit,
        'sl_hit_time_secs': sl_hit_time,
        'net_pnl': net_pnl,
        'adx': crossover['adx'],
        'dmi_spread': crossover['dmi_spread'],
        'dmi_plus': crossover['dmi_plus'],
        'dmi_minus': crossover['dmi_minus'],
    }


def print_report(results, sl_ticks, lookahead_secs):
    """Print comprehensive DMI crossover validation report."""
    if not results:
        print("No results to report.")
        return

    df = pd.DataFrame(results)
    n = len(df)

    print(f"\n{'='*70}")
    print(f"  DMI CROSSOVER VALIDATION REPORT")
    print(f"{'='*70}")
    print(f"  Total crossovers analyzed: {n}")
    print(f"  Lookahead window:          {lookahead_secs}s ({lookahead_secs/60:.0f} min)")
    print(f"  Stop loss:                 {sl_ticks} ticks (${sl_ticks * TICK_SIZE * POINT_VALUE:.0f})")
    print(f"  LONG signals:              {len(df[df['direction']=='LONG'])}")
    print(f"  SHORT signals:             {len(df[df['direction']=='SHORT'])}")

    # -- Overall outcomes --
    wins = df[~df['sl_hit']]
    losses = df[df['sl_hit']]
    wr = len(wins) / n * 100 if n > 0 else 0

    print(f"\n-- OVERALL OUTCOMES (${sl_ticks * TICK_SIZE * POINT_VALUE:.0f} SL) --")
    print(f"  Win rate:     {wr:.1f}% ({len(wins)}/{n})")
    print(f"  Avg MFE:      {df['mfe_ticks'].mean():.1f} ticks (${df['mfe_dollars'].mean():.2f})")
    print(f"  Median MFE:   {df['mfe_ticks'].median():.1f} ticks (${df['mfe_dollars'].median():.2f})")
    print(f"  Avg MAE:      {df['mae_before_mfe_ticks'].mean():.1f} ticks (${df['mae_before_mfe_dollars'].mean():.2f})")
    print(f"  Avg net PnL:  ${df['net_pnl'].mean():.2f}")
    print(f"  Total net:    ${df['net_pnl'].sum():.2f}")

    # -- Win stats --
    if len(wins) > 0:
        print(f"\n-- WINNING TRADES --")
        print(f"  Avg MFE:         {wins['mfe_ticks'].mean():.1f} ticks (${wins['mfe_dollars'].mean():.2f})")
        print(f"  Median MFE:      {wins['mfe_ticks'].median():.1f} ticks")
        print(f"  Avg time to MFE: {wins['time_to_mfe_secs'].mean():.0f}s ({wins['time_to_mfe_secs'].mean()/60:.1f} min)")
        print(f"  Median time:     {wins['time_to_mfe_secs'].median():.0f}s ({wins['time_to_mfe_secs'].median()/60:.1f} min)")
        print(f"  Avg MAE before:  {wins['mae_before_mfe_ticks'].mean():.1f} ticks (${wins['mae_before_mfe_dollars'].mean():.2f})")

    # -- Loss stats --
    if len(losses) > 0:
        print(f"\n-- LOSING TRADES (SL HIT) --")
        print(f"  Count:           {len(losses)}")
        print(f"  Avg SL hit time: {losses['sl_hit_time_secs'].dropna().mean():.0f}s")
        print(f"  Avg MFE before:  {losses['mfe_ticks'].mean():.1f} ticks (max favorable before stopped)")

    # -- Direction breakdown --
    print(f"\n-- DIRECTION BREAKDOWN --")
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0:
            continue
        sub_wins = sub[~sub['sl_hit']]
        sub_wr = len(sub_wins) / len(sub) * 100
        print(f"  {d}: {len(sub)} signals, {sub_wr:.1f}% WR, "
              f"avg MFE={sub['mfe_ticks'].mean():.1f}t, "
              f"avg net=${sub['net_pnl'].mean():.2f}, "
              f"total=${sub['net_pnl'].sum():.2f}")

    # -- MFE buckets --
    print(f"\n-- MFE DISTRIBUTION (all crossovers, before SL) --")
    thresholds = [0, 4, 8, 20, 40, 60, 100, 200]
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        count = len(df[(df['mfe_ticks'] >= lo) & (df['mfe_ticks'] < hi)])
        pct = count / n * 100
        lo_d = lo * TICK_SIZE * POINT_VALUE
        hi_d = hi * TICK_SIZE * POINT_VALUE
        print(f"  {lo:>4}-{hi:<4} ticks (${lo_d:>5.0f}-${hi_d:<5.0f}): "
              f"{count:>5} ({pct:>5.1f}%)")
    count_big = len(df[df['mfe_ticks'] >= thresholds[-1]])
    print(f"  {thresholds[-1]:>4}+    ticks (${thresholds[-1]*TICK_SIZE*POINT_VALUE:>5.0f}+     ): "
          f"{count_big:>5} ({count_big/n*100:>5.1f}%)")

    # -- ADX influence --
    print(f"\n-- ADX AT CROSSOVER --")
    adx_bins = [(0, 15), (15, 25), (25, 35), (35, 50), (50, 100)]
    for lo, hi in adx_bins:
        sub = df[(df['adx'] >= lo) & (df['adx'] < hi)]
        if len(sub) == 0:
            continue
        sub_wins = sub[~sub['sl_hit']]
        sub_wr = len(sub_wins) / len(sub) * 100
        print(f"  ADX {lo:>2}-{hi:<3}: {len(sub):>5} signals, "
              f"{sub_wr:.1f}% WR, "
              f"avg MFE={sub['mfe_ticks'].mean():.1f}t, "
              f"avg net=${sub['net_pnl'].mean():.2f}")

    # -- DMI spread influence --
    print(f"\n-- DMI SPREAD AT CROSSOVER (|DI+ - DI-|) --")
    spread_bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
    for lo, hi in spread_bins:
        sub = df[(df['dmi_spread'] >= lo) & (df['dmi_spread'] < hi)]
        if len(sub) == 0:
            continue
        sub_wins = sub[~sub['sl_hit']]
        sub_wr = len(sub_wins) / len(sub) * 100
        print(f"  Spread {lo:>2}-{hi:<3}: {len(sub):>5} signals, "
              f"{sub_wr:.1f}% WR, "
              f"avg MFE={sub['mfe_ticks'].mean():.1f}t, "
              f"avg net=${sub['net_pnl'].mean():.2f}")

    # -- Risk/Reward summary --
    survivors = df[~df['sl_hit']]
    if len(survivors) > 0:
        avg_reward = survivors['mfe_dollars'].mean()
        sl_cost = sl_ticks * TICK_SIZE * POINT_VALUE
        rr = avg_reward / sl_cost if sl_cost > 0 else 0
        ev = (wr / 100) * avg_reward - ((100 - wr) / 100) * sl_cost
        print(f"\n-- RISK/REWARD --")
        print(f"  Risk:          ${sl_cost:.0f}")
        print(f"  Avg reward:    ${avg_reward:.2f}")
        print(f"  R:R ratio:     1:{rr:.1f}")
        print(f"  EV per trade:  ${ev:.2f}")
        print(f"  Expected net:  ${ev * n:.2f} ({n} trades)")


def plot_results(results, output_path):
    """4-panel visualization of DMI crossover outcomes."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DMI Crossover Validation', fontsize=16, fontweight='bold')

    # Panel 1: MFE vs MAE scatter
    ax = axes[0, 0]
    colors = ['#2ecc71' if not sl else '#e74c3c' for sl in df['sl_hit']]
    ax.scatter(df['mae_before_mfe_ticks'], df['mfe_ticks'],
               c=colors, alpha=0.3, s=10, edgecolors='none')
    ax.axhline(y=SL_TICKS, color='orange', linestyle='--', alpha=0.7, label=f'SL = {SL_TICKS}t')
    ax.axvline(x=SL_TICKS, color='red', linestyle='--', alpha=0.7, label=f'MAE = SL')
    ax.set_xlabel('MAE before MFE (ticks)')
    ax.set_ylabel('MFE (ticks)')
    ax.set_title('MFE vs MAE (green=survived, red=stopped)')
    ax.legend(fontsize=9)

    # Panel 2: MFE histogram
    ax = axes[0, 1]
    ax.hist(df['mfe_ticks'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=df['mfe_ticks'].mean(), color='red', linestyle='--',
               label=f'Mean={df["mfe_ticks"].mean():.0f}t')
    ax.axvline(x=df['mfe_ticks'].median(), color='orange', linestyle='--',
               label=f'Median={df["mfe_ticks"].median():.0f}t')
    ax.set_xlabel('MFE (ticks)')
    ax.set_ylabel('Count')
    ax.set_title('MFE Distribution')
    ax.legend(fontsize=9)

    # Panel 3: Time to MFE
    ax = axes[1, 0]
    survived = df[~df['sl_hit']]
    if len(survived) > 0:
        ax.hist(survived['time_to_mfe_secs'] / 60, bins=50,
                color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(x=survived['time_to_mfe_secs'].median() / 60,
                   color='orange', linestyle='--',
                   label=f'Median={survived["time_to_mfe_secs"].median()/60:.1f}min')
    ax.set_xlabel('Time to MFE (minutes)')
    ax.set_ylabel('Count')
    ax.set_title('Time to MFE (surviving trades)')
    ax.legend(fontsize=9)

    # Panel 4: ADX vs Win Rate
    ax = axes[1, 1]
    adx_edges = np.arange(0, 55, 5)
    wr_by_adx = []
    mfe_by_adx = []
    adx_centers = []
    for i in range(len(adx_edges) - 1):
        sub = df[(df['adx'] >= adx_edges[i]) & (df['adx'] < adx_edges[i + 1])]
        if len(sub) >= 5:
            wr_by_adx.append(len(sub[~sub['sl_hit']]) / len(sub) * 100)
            mfe_by_adx.append(sub['mfe_ticks'].mean())
            adx_centers.append((adx_edges[i] + adx_edges[i + 1]) / 2)

    ax2 = ax.twinx()
    if adx_centers:
        ax.bar(adx_centers, wr_by_adx, width=4, color='steelblue', alpha=0.6, label='Win Rate')
        ax2.plot(adx_centers, mfe_by_adx, 'o-', color='orange', label='Avg MFE')
    ax.set_xlabel('ADX at Crossover')
    ax.set_ylabel('Win Rate %', color='steelblue')
    ax2.set_ylabel('Avg MFE (ticks)', color='orange')
    ax.set_title('ADX vs Win Rate & MFE')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='DMI Crossover Validation')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS data directory')
    parser.add_argument('--months', nargs='*', default=None,
                        help='Specific months (e.g., 2025_07 2025_08). Default=all')
    parser.add_argument('--lookahead', type=int, default=LOOKAHEAD_SECS,
                        help=f'Lookahead seconds after crossover (default {LOOKAHEAD_SECS})')
    parser.add_argument('--sl', type=int, default=SL_TICKS,
                        help=f'Stop loss in ticks (default {SL_TICKS})')
    parser.add_argument('--min-adx', type=float, default=MIN_ADX,
                        help='Minimum ADX at crossover (default 0 = no filter)')
    parser.add_argument('--output', default=None,
                        help='Output plot path')
    args = parser.parse_args()

    print(f"DMI Crossover Validation")
    print(f"  Data: {args.data_dir}")
    print(f"  Lookahead: {args.lookahead}s ({args.lookahead/60:.0f} min)")
    print(f"  SL: {args.sl} ticks (${args.sl * TICK_SIZE * POINT_VALUE:.0f})")
    print(f"  Min ADX: {args.min_adx}")

    # Load 1m data
    print(f"\nLoading 1m data...")
    tf_dir = os.path.join(args.data_dir, '1m')
    if not os.path.isdir(tf_dir):
        print(f"ERROR: 1m directory not found at {tf_dir}")
        sys.exit(1)

    import glob as globmod
    if args.months:
        files = [os.path.join(tf_dir, f'{m}.parquet') for m in args.months]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(globmod.glob(os.path.join(tf_dir, '*.parquet')))

    if not files:
        print(f"ERROR: No parquet files found in {tf_dir}")
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in files]
    df_1m = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in df_1m.columns:
        if pd.api.types.is_datetime64_any_dtype(df_1m['timestamp']):
            df_1m['timestamp'] = df_1m['timestamp'].astype('int64') // 10**9
        df_1m = df_1m.sort_values('timestamp').reset_index(drop=True)

    print(f"  Loaded {len(df_1m)} bars from {len(files)} files")

    # Compute DMI
    print(f"\nComputing DMI (period=14)...")
    df_dmi = compute_dmi_from_ohlc(df_1m, period=14)

    # Skip warmup (first 28 bars minimum for ADX to stabilize)
    df_dmi = df_dmi.iloc[28:].reset_index(drop=True)
    print(f"  DMI computed for {len(df_dmi)} bars (after warmup)")

    # Detect crossovers
    print(f"\nDetecting DI+/DI- crossovers...")
    crossovers = detect_crossovers(df_dmi, min_adx=args.min_adx)
    n_long = sum(1 for c in crossovers if c['direction'] == 'LONG')
    n_short = sum(1 for c in crossovers if c['direction'] == 'SHORT')
    print(f"  Found {len(crossovers)} crossovers ({n_long} LONG, {n_short} SHORT)")

    if not crossovers:
        print("No crossovers found. Try lowering --min-adx.")
        return

    # Load 1s index for outcome measurement
    print(f"\nLoading 1s data index...")
    index_1s = load_1s_index(args.data_dir)

    # Measure outcomes
    print(f"\nMeasuring crossover outcomes...")
    cache = {}
    results = []
    skipped = 0

    for xover in tqdm(crossovers, desc='Validating crossovers', unit='xover',
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                 '[{elapsed}<{remaining}]'):
        outcome = measure_crossover_outcome(xover, index_1s, args.lookahead, args.sl, cache)
        if outcome is None:
            skipped += 1
            continue
        results.append(outcome)

    print(f"  Analyzed: {len(results)}, Skipped (no 1s data): {skipped}")

    # Report
    print_report(results, args.sl, args.lookahead)

    # Plot
    if results:
        out_path = args.output or 'tools/plots/standalone/dmi_crossover/dmi_validation.png'
        plot_results(results, out_path)

        # Save CSV
        csv_path = out_path.replace('.png', '.csv')
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Golden Path Metrics — Y10, Y11, Y12 computation.

Computes per-trade golden path metrics using 1s ATLAS data:
  Y10: actual_pnl / chord_length  (capture efficiency vs theoretical max)
  Y11: oracle optimal segments    (best risk-adjusted extraction benchmark)
  Y12: actual_pnl / oracle_mae    (risk-adjusted path ratio)

Usage:
    python tools/golden_path.py                              # IS trades
    python tools/golden_path.py --dir reports/oos            # OOS trades
    python tools/golden_path.py --data-dir DATA/ATLAS_OOS    # custom 1s data
    python tools/golden_path.py --output reports/is/golden_path.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_1s_index(data_dir: str) -> dict:
    """Load 1s parquet files into a dict keyed by year_month for lazy loading."""
    p = Path(data_dir) / '1s'
    if not p.exists():
        print(f"ERROR: 1s data not found at {p}")
        sys.exit(1)

    files = sorted(p.glob('*.parquet'))
    index = {}
    for f in files:
        # Extract year_month from filename like 2025_01.parquet
        stem = f.stem  # '2025_01'
        index[stem] = f
    print(f"  Found {len(index)} 1s parquet files in {p}")
    return index


def load_1s_window(index: dict, ts_start: float, ts_end: float,
                   _cache: dict = {}) -> pd.DataFrame:
    """Load 1s bars for a time window, with caching."""
    # Determine which files we need
    import datetime
    dt_start = datetime.datetime.utcfromtimestamp(ts_start)
    dt_end = datetime.datetime.utcfromtimestamp(ts_end)

    needed = set()
    cur = dt_start.replace(day=1)
    while cur <= dt_end:
        key = f"{cur.year}_{cur.month:02d}"
        if key in index:
            needed.add(key)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    frames = []
    for key in sorted(needed):
        if key not in _cache:
            _cache[key] = pd.read_parquet(index[key])
        df = _cache[key]
        mask = (df['timestamp'] >= ts_start) & (df['timestamp'] <= ts_end)
        frames.append(df[mask])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values('timestamp')


def chord_length(prices: np.ndarray) -> float:
    """Sum of |delta| at 1s resolution — the theoretical maximum extraction."""
    if len(prices) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(prices))))


def oracle_segments(prices: np.ndarray, tick_size: float = 0.25,
                    min_run_ticks: float = 2.0, mae_tolerance_ticks: float = 3.0):
    """
    Segment 1s prices into optimal directional runs.

    A run continues as long as:
    - The adverse excursion from run start stays below mae_tolerance
    - OR the favorable excursion keeps growing

    Returns dict with total captured ticks, total MAE, segment count.
    """
    if len(prices) < 3:
        return {'captured_ticks': 0, 'total_mae': 0, 'n_segments': 0,
                'efficiency': 0}

    segments = []
    i = 0
    n = len(prices)

    while i < n - 1:
        entry = prices[i]
        best_favorable = 0.0
        worst_adverse = 0.0
        best_j = i

        # Try long
        long_captured = 0
        long_mae = 0
        long_end = i
        for j in range(i + 1, n):
            delta = (prices[j] - entry) / tick_size
            fav = max(0, delta)
            adv = max(0, -delta)
            if fav > long_captured:
                long_captured = fav
                long_end = j
                long_mae = max(long_mae, adv)
            if adv > mae_tolerance_ticks and fav < min_run_ticks:
                break
            if adv > mae_tolerance_ticks * 2:
                break

        # Try short
        short_captured = 0
        short_mae = 0
        short_end = i
        for j in range(i + 1, n):
            delta = (entry - prices[j]) / tick_size
            fav = max(0, delta)
            adv = max(0, -delta)
            if fav > short_captured:
                short_captured = fav
                short_end = j
                short_mae = max(short_mae, adv)
            if adv > mae_tolerance_ticks and fav < min_run_ticks:
                break
            if adv > mae_tolerance_ticks * 2:
                break

        # Pick the better direction
        if long_captured >= short_captured and long_captured >= min_run_ticks:
            segments.append({
                'captured': long_captured,
                'mae': long_mae,
                'bars': long_end - i,
            })
            i = long_end + 1
        elif short_captured >= min_run_ticks:
            segments.append({
                'captured': short_captured,
                'mae': short_mae,
                'bars': short_end - i,
            })
            i = short_end + 1
        else:
            i += 1  # skip this bar, no viable run

    total_captured = sum(s['captured'] for s in segments)
    total_mae = sum(s['mae'] for s in segments) if segments else 0
    efficiency = total_captured / total_mae if total_mae > 0 else float('inf')

    return {
        'captured_ticks': round(total_captured, 2),
        'total_mae': round(total_mae, 2),
        'n_segments': len(segments),
        'efficiency': round(efficiency, 2),
    }


def compute_golden_path(trade_log: pd.DataFrame, data_dir: str,
                        tick_size: float = 0.25) -> pd.DataFrame:
    """Add Y10, Y11, Y12 columns to the trade log."""
    index = load_1s_index(data_dir)

    y10_list = []  # chord capture efficiency
    y11_list = []  # oracle path efficiency
    y12_list = []  # risk-adjusted ratio

    cache = {}

    for _, row in tqdm(trade_log.iterrows(), total=len(trade_log),
                       desc='Computing golden path', unit='trade',
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                  '[{elapsed}<{remaining}]'):
        ts_entry = float(row['entry_time'])
        ts_exit = float(row['exit_time'])
        actual_pnl = float(row['actual_pnl'])
        oracle_mae = float(row.get('oracle_mae', 0))

        # Load 1s window
        window = load_1s_window(index, ts_entry, ts_exit, cache)

        if len(window) < 2:
            y10_list.append(np.nan)
            y11_list.append(np.nan)
            y12_list.append(actual_pnl / oracle_mae if oracle_mae > 0 else np.nan)
            continue

        prices = window['close'].values

        # Y10: actual / chord length
        cl = chord_length(prices)
        cl_ticks = cl / tick_size
        y10 = actual_pnl / cl_ticks if cl_ticks > 0 else 0.0
        y10_list.append(round(y10, 4))

        # Y11: oracle optimal segmentation
        seg = oracle_segments(prices, tick_size=tick_size)
        y11_list.append(seg['efficiency'])

        # Y12: actual / MAE
        y12 = actual_pnl / oracle_mae if oracle_mae > 0 else np.nan
        y12_list.append(round(y12, 4))

    trade_log = trade_log.copy()
    trade_log['Y10_chord_capture'] = y10_list
    trade_log['Y11_oracle_efficiency'] = y11_list
    trade_log['Y12_risk_adj'] = y12_list

    return trade_log


def print_summary(df: pd.DataFrame):
    """Print golden path summary statistics."""
    print(f"\n{'='*60}")
    print("GOLDEN PATH SUMMARY")
    print(f"{'='*60}")
    print(f"Trades analyzed: {len(df)}")

    for col, name, desc in [
        ('Y10_chord_capture', 'Y10 Chord Capture',
         'actual_pnl / chord_length (% of max extraction)'),
        ('Y11_oracle_efficiency', 'Y11 Oracle Efficiency',
         'oracle_captured / oracle_MAE (benchmark ratio)'),
        ('Y12_risk_adj', 'Y12 Risk-Adjusted',
         'actual_pnl / oracle_MAE (reward per risk)'),
    ]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            print(f"\n  {name}: no data")
            continue

        print(f"\n  {name} ({desc}):")
        print(f"    N      = {len(s)}")
        print(f"    Mean   = {s.mean():.4f}")
        print(f"    Median = {s.median():.4f}")
        print(f"    Std    = {s.std():.4f}")
        for p in [10, 25, 75, 90]:
            print(f"    p{p:<2}    = {np.percentile(s, p):.4f}")

    # Level 0 → 1 → 2 gap analysis
    if 'Y10_chord_capture' in df.columns and 'Y11_oracle_efficiency' in df.columns:
        y10 = df['Y10_chord_capture'].dropna()
        y11 = df['Y11_oracle_efficiency'].dropna()
        if len(y10) > 0 and len(y11) > 0:
            print(f"\n  GAP ANALYSIS:")
            print(f"    Level 0 (chord): 100% (theoretical max)")
            print(f"    Level 2 (us):    {y10.mean()*100:.1f}% capture")
            gap = 100.0 - y10.mean() * 100
            print(f"    Total gap:       {gap:.1f}%")

    # Win/Loss breakdown
    if 'result' in df.columns and 'Y10_chord_capture' in df.columns:
        for label in ['WIN', 'LOSS']:
            sub = df[df['result'] == label]['Y10_chord_capture'].dropna()
            if len(sub) > 0:
                print(f"\n  {label} trades Y10: mean={sub.mean():.4f} "
                      f"median={sub.median():.4f} N={len(sub)}")


def main():
    parser = argparse.ArgumentParser(description='Golden Path Metrics (Y10/Y11/Y12)')
    parser.add_argument('--dir', default='reports/is',
                        help='Directory with oracle_trade_log.csv')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS data directory (must contain 1s/ subfolder)')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: same dir as input)')
    parser.add_argument('--tick-size', type=float, default=0.25,
                        help='Tick size (MNQ=0.25)')
    args = parser.parse_args()

    # Load trade log
    log_path = None
    for name in ('oracle_trade_log.csv', 'is_trade_log.csv', 'oos_trade_log.csv'):
        p = os.path.join(args.dir, name)
        if os.path.exists(p):
            log_path = p
            break
    if log_path is None:
        print(f"ERROR: No trade log found in {args.dir}")
        sys.exit(1)

    df = pd.read_csv(log_path)
    print(f"Loaded {log_path}: {len(df)} trades")

    # Compute golden path metrics
    df = compute_golden_path(df, args.data_dir, tick_size=args.tick_size)

    # Save
    out = args.output or os.path.join(args.dir, 'golden_path.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # Summary
    print_summary(df)


if __name__ == '__main__':
    main()

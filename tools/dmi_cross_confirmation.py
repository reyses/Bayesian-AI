"""
DMI Crossover Confirmation Lag Research
========================================
How many bars after a 5m DMI crossover does it take to confirm
a true direction change? Measures accuracy at bar 0, 1, 2, ... N
after the cross event.

A "true" direction change = price moved >= X ticks in the cross direction
within the next 30 bars (7.5 minutes at 15s).

Usage:
    python tools/dmi_cross_confirmation.py --data DATA/ATLAS --month 2025_06
    python tools/dmi_cross_confirmation.py --data DATA/ATLAS_1WEEK
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tf_data(data_dir: str, tf: str, month: str = None):
    tf_dir = os.path.join(data_dir, tf)
    files = sorted(Path(tf_dir).glob('*.parquet'))
    if month:
        files = [f for f in files if month in f.stem]
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)


def run_dmi_confirmation(data_dir: str, month: str = None):
    from core.statistical_field_engine import StatisticalFieldEngine

    engine = StatisticalFieldEngine(regression_period=21, use_gpu=True)

    print("=" * 70)
    print("  DMI CROSSOVER CONFIRMATION LAG RESEARCH")
    print(f"  Data: {data_dir}" + (f" (month={month})" if month else ""))
    print("=" * 70)

    # Test on multiple TFs
    test_tfs = ['5m', '3m', '1m']
    # Use 15s as execution TF for outcome measurement
    print("\n[1] Loading 15s execution data...")
    df_15s = load_tf_data(data_dir, '15s', month)
    prices_15s = df_15s['close'].values
    ts_15s = df_15s['timestamp'].values
    print(f"  {len(df_15s):,} 15s bars")

    for tf in test_tfs:
        print(f"\n{'='*70}")
        print(f"  ANALYZING {tf} DMI CROSSOVERS")
        print(f"{'='*70}")

        print(f"\n[2] Loading {tf} data...")
        df_tf = load_tf_data(data_dir, tf, month)
        print(f"  {len(df_tf):,} {tf} bars")

        print(f"\n[3] Computing {tf} market states...")
        states = engine.batch_compute_states(df_tf, use_cuda=True)
        print(f"  {len(states):,} states")

        # Extract DMI series
        dmi_plus = np.array([s['state'].dmi_plus for s in states])
        dmi_minus = np.array([s['state'].dmi_minus for s in states])
        state_ts = np.array([s['state'].timestamp for s in states])

        # Find crossover events
        # Cross LONG: dmi_plus crosses above dmi_minus
        # Cross SHORT: dmi_minus crosses above dmi_plus
        crosses = []
        for i in range(1, len(dmi_plus)):
            gap = abs(dmi_plus[i] - dmi_minus[i])
            prev_long = dmi_plus[i-1] > dmi_minus[i-1]
            curr_long = dmi_plus[i] > dmi_minus[i]

            if prev_long != curr_long:
                direction = 'LONG' if curr_long else 'SHORT'
                crosses.append({
                    'bar_idx': i,
                    'timestamp': state_ts[i],
                    'direction': direction,
                    'gap': gap,
                    'dmi_plus': dmi_plus[i],
                    'dmi_minus': dmi_minus[i],
                })

        print(f"\n[4] Found {len(crosses):,} DMI crossovers")

        # For each gap threshold, measure confirmation accuracy at different lags
        gap_thresholds = [0, 3, 5, 8]
        confirmation_windows = [0, 1, 2, 3, 4, 5, 8, 10, 15, 20]
        outcome_window = 40  # 10 minutes at 15s — did price move in cross direction?
        outcome_threshold_ticks = 8  # 4 ticks = $2 — minimum "real" move

        print(f"\n[5] Measuring confirmation accuracy...")
        print(f"  Outcome: price moves >={outcome_threshold_ticks} ticks in cross direction")
        print(f"  within {outcome_window} 15s bars ({outcome_window*15/60:.0f} min)")

        for gap_min in gap_thresholds:
            filtered = [c for c in crosses if c['gap'] >= gap_min]
            if len(filtered) < 20:
                print(f"\n  Gap>={gap_min}: {len(filtered)} crosses (too few)")
                continue

            print(f"\n  Gap>={gap_min}: {len(filtered)} crosses")
            print(f"  {'Confirm bars':>14s}  {'Survived':>9s}  {'Accurate':>9s}  {'Accuracy':>9s}  {'Avg MFE':>9s}  {'Avg MAE':>9s}")
            print(f"  {'-'*65}")

            for confirm_bars in confirmation_windows:
                accurate = 0
                total = 0
                mfe_list = []
                mae_list = []

                for cross in filtered:
                    cross_ts = cross['timestamp']
                    cross_bar = cross['bar_idx']

                    # Check if DMI still agrees after confirm_bars
                    if confirm_bars > 0 and cross_bar + confirm_bars < len(dmi_plus):
                        still_long = dmi_plus[cross_bar + confirm_bars] > dmi_minus[cross_bar + confirm_bars]
                        expected_long = (cross['direction'] == 'LONG')
                        if still_long != expected_long:
                            continue  # DMI reversed — this cross didn't confirm

                    # Find matching 15s bar for outcome measurement
                    idx_15s = np.searchsorted(ts_15s, cross_ts)
                    if idx_15s >= len(ts_15s) - outcome_window:
                        continue

                    # Measure outcome on 15s bars
                    entry_price = prices_15s[idx_15s]
                    future = prices_15s[idx_15s+1 : idx_15s+1+outcome_window]

                    if cross['direction'] == 'LONG':
                        moves = future - entry_price
                    else:
                        moves = entry_price - future

                    mfe = moves.max() / 0.25  # ticks
                    mae = (-moves).max() / 0.25  # ticks

                    total += 1
                    if mfe >= outcome_threshold_ticks:
                        accurate += 1
                    mfe_list.append(mfe)
                    mae_list.append(mae)

                if total < 10:
                    continue

                acc = accurate / total * 100
                avg_mfe = np.mean(mfe_list)
                avg_mae = np.mean(mae_list)
                print(f"  {confirm_bars:>10d} bar  {total:>9d}  {accurate:>9d}  {acc:>8.1f}%  {avg_mfe:>8.1f}t  {avg_mae:>8.1f}t")

        # ADX strength at crossover vs accuracy
        print(f"\n[6] ADX at crossover vs accuracy (gap>=5)")
        filtered = [c for c in crosses if c['gap'] >= 5]
        if len(filtered) >= 20:
            # Get ADX at each cross
            for cross in filtered:
                cross['adx'] = states[cross['bar_idx']]['state'].adx_strength

            # Bucket by ADX
            adx_vals = [c['adx'] for c in filtered]
            adx_buckets = [(0, 20, 'ADX<20 (weak)'), (20, 30, 'ADX 20-30'),
                           (30, 45, 'ADX 30-45'), (45, 100, 'ADX 45+')]

            print(f"  {'ADX bucket':>20s}  {'Crosses':>8s}  {'Accurate':>9s}  {'Accuracy':>9s}")
            print(f"  {'-'*52}")

            for lo, hi, label in adx_buckets:
                bucket = [c for c in filtered if lo <= c['adx'] < hi]
                if len(bucket) < 5:
                    continue

                acc_count = 0
                for cross in bucket:
                    idx_15s = np.searchsorted(ts_15s, cross['timestamp'])
                    if idx_15s >= len(ts_15s) - outcome_window:
                        continue
                    entry = prices_15s[idx_15s]
                    future = prices_15s[idx_15s+1:idx_15s+1+outcome_window]
                    if cross['direction'] == 'LONG':
                        mfe = (future - entry).max() / 0.25
                    else:
                        mfe = (entry - future).max() / 0.25
                    if mfe >= outcome_threshold_ticks:
                        acc_count += 1

                acc = acc_count / len(bucket) * 100
                print(f"  {label:>20s}  {len(bucket):>8d}  {acc_count:>9d}  {acc:>8.1f}%")

    # Save report
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"reports/findings/dmi_confirmation_{ts_str}.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"DMI Crossover Confirmation Research — {datetime.now()}\n")
        f.write(f"Data: {data_dir} (month={month})\n")
        f.write(f"See terminal output for full results\n")
    print(f"\n  Report saved: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS_1WEEK')
    parser.add_argument('--month', default=None)
    args = parser.parse_args()
    run_dmi_confirmation(args.data, args.month)

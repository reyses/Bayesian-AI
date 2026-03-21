"""
DMI Sample Size Research
========================
For each TF, what ADX_PERIOD produces the most reliable DMI crossover?
Tests period 7, 10, 14, 21, 30, 50 on each TF and measures crossover
accuracy (did price move in cross direction within N bars?).

Usage:
    python tools/dmi_sample_size_research.py --data DATA/ATLAS --month 2025_06
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_dmi_custom(highs, lows, closes, period: int):
    """Compute DMI+/- with custom period (Wilder smoothing)."""
    n = len(highs)
    if n < period + 1:
        return np.zeros(n), np.zeros(n)

    # TR, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)

        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0

    # Wilder smoothing
    dmi_plus = np.zeros(n)
    dmi_minus = np.zeros(n)

    smooth_tr = np.sum(tr[1:period+1])
    smooth_plus = np.sum(plus_dm[1:period+1])
    smooth_minus = np.sum(minus_dm[1:period+1])

    if smooth_tr > 0:
        dmi_plus[period] = 100.0 * smooth_plus / smooth_tr
        dmi_minus[period] = 100.0 * smooth_minus / smooth_tr

    for i in range(period + 1, n):
        smooth_tr = smooth_tr - smooth_tr / period + tr[i]
        smooth_plus = smooth_plus - smooth_plus / period + plus_dm[i]
        smooth_minus = smooth_minus - smooth_minus / period + minus_dm[i]
        if smooth_tr > 0:
            dmi_plus[i] = 100.0 * smooth_plus / smooth_tr
            dmi_minus[i] = 100.0 * smooth_minus / smooth_tr

    return dmi_plus, dmi_minus


def measure_crossover_accuracy(dmi_plus, dmi_minus, prices, gap_min: float,
                                outcome_bars: int, outcome_ticks: float,
                                tick_size: float = 0.25):
    """Measure accuracy of DMI crossovers."""
    n = len(dmi_plus)
    accurate = 0
    total = 0
    mfe_list = []
    mae_list = []

    for i in range(1, n - outcome_bars):
        prev_long = dmi_plus[i-1] > dmi_minus[i-1]
        curr_long = dmi_plus[i] > dmi_minus[i]

        if prev_long == curr_long:
            continue  # no cross

        gap = abs(dmi_plus[i] - dmi_minus[i])
        if gap < gap_min:
            continue

        direction = 1 if curr_long else -1
        future = prices[i+1:i+1+outcome_bars]
        moves = (future - prices[i]) * direction / tick_size  # ticks

        mfe = moves.max()
        mae = (-moves).max()
        mfe_list.append(mfe)
        mae_list.append(mae)
        total += 1
        if mfe >= outcome_ticks:
            accurate += 1

    if total < 5:
        return None

    return {
        'total': total,
        'accurate': accurate,
        'accuracy': accurate / total * 100,
        'avg_mfe': np.mean(mfe_list),
        'avg_mae': np.mean(mae_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--month', default='2025_06')
    args = parser.parse_args()

    test_tfs = ['1s', '5s', '15s', '30s', '1m', '3m', '5m', '15m']
    test_periods = [7, 10, 14, 21, 30, 50]
    gap_min = 3.0
    outcome_ticks = 8.0  # minimum "real" move

    # Load 15s for outcome measurement
    print("Loading 15s execution data...")
    tf_dir = os.path.join(args.data, '15s')
    files = sorted(Path(tf_dir).glob('*.parquet'))
    if args.month:
        files = [f for f in files if args.month in f.stem]
    df_15s = pd.concat([pd.read_parquet(f) for f in files]).sort_values('timestamp').reset_index(drop=True)
    prices_15s = df_15s['close'].values
    ts_15s = df_15s['timestamp'].values
    print(f"  {len(df_15s):,} 15s bars")

    # Outcome window: 40 bars at 15s = 10 min
    outcome_bars_15s = 40

    print(f"\nDMI SAMPLE SIZE RESEARCH")
    print(f"  Gap >= {gap_min}, outcome >= {outcome_ticks} ticks in {outcome_bars_15s} 15s bars")
    print(f"  Data: {args.data} month={args.month}")
    print("=" * 90)

    results = []

    for tf in test_tfs:
        tf_dir_path = os.path.join(args.data, tf)
        if not os.path.isdir(tf_dir_path):
            continue

        files = sorted(Path(tf_dir_path).glob('*.parquet'))
        if args.month:
            files = [f for f in files if args.month in f.stem]
        if not files:
            continue

        df_tf = pd.concat([pd.read_parquet(f) for f in files]).sort_values('timestamp').reset_index(drop=True)
        highs = df_tf['high'].values
        lows = df_tf['low'].values
        closes = df_tf['close'].values
        ts_tf = df_tf['timestamp'].values

        tf_sec_map = {'1s': 1, '5s': 5, '15s': 15, '30s': 30, '1m': 60,
                      '3m': 180, '5m': 300, '15m': 900}
        tf_sec = tf_sec_map.get(tf, 15)

        print(f"\n  {tf} ({len(df_tf):,} bars)")
        print(f"  {'Period':>8s}  {'Sample Win':>11s}  {'Crosses':>8s}  {'Accuracy':>9s}  "
              f"{'Avg MFE':>9s}  {'Avg MAE':>9s}  {'MFE/MAE':>8s}")
        print(f"  {'-'*65}")

        for period in test_periods:
            sample_window_sec = period * tf_sec
            sample_window_min = sample_window_sec / 60

            dmi_plus, dmi_minus = compute_dmi_custom(highs, lows, closes, period)

            # Find crosses and measure on 15s outcome
            accurate = 0
            total = 0
            mfe_list = []
            mae_list = []

            for i in range(period + 1, len(dmi_plus)):
                prev_long = dmi_plus[i-1] > dmi_minus[i-1]
                curr_long = dmi_plus[i] > dmi_minus[i]
                if prev_long == curr_long:
                    continue
                gap = abs(dmi_plus[i] - dmi_minus[i])
                if gap < gap_min:
                    continue

                # Map to 15s for outcome
                cross_ts = ts_tf[i]
                idx_15s = np.searchsorted(ts_15s, cross_ts)
                if idx_15s >= len(ts_15s) - outcome_bars_15s:
                    continue

                entry = prices_15s[idx_15s]
                future = prices_15s[idx_15s+1:idx_15s+1+outcome_bars_15s]
                direction = 1 if curr_long else -1
                moves = (future - entry) * direction / 0.25

                mfe = moves.max()
                mae = (-moves).max()
                mfe_list.append(mfe)
                mae_list.append(mae)
                total += 1
                if mfe >= outcome_ticks:
                    accurate += 1

            if total < 5:
                print(f"  {period:>8d}  {sample_window_min:>8.1f}min  {total:>8d}  {'(too few)':>9s}")
                continue

            acc = accurate / total * 100
            avg_mfe = np.mean(mfe_list)
            avg_mae = np.mean(mae_list)
            ratio = avg_mfe / avg_mae if avg_mae > 0 else 0

            marker = ' <-- best' if acc > 95 and ratio > 3 else ''
            print(f"  {period:>8d}  {sample_window_min:>8.1f}min  {total:>8d}  {acc:>8.1f}%  "
                  f"{avg_mfe:>8.1f}t  {avg_mae:>8.1f}t  {ratio:>7.1f}x{marker}")

            results.append({
                'tf': tf, 'period': period, 'sample_min': sample_window_min,
                'crosses': total, 'accuracy': acc,
                'avg_mfe': avg_mfe, 'avg_mae': avg_mae, 'ratio': ratio,
            })

    # Summary: best period per TF
    if results:
        rdf = pd.DataFrame(results)
        print(f"\n{'='*90}")
        print(f"BEST PERIOD PER TF (highest accuracy with MFE/MAE > 2):")
        print(f"{'='*90}")
        for tf in test_tfs:
            sub = rdf[(rdf['tf'] == tf) & (rdf['ratio'] > 2)]
            if len(sub) == 0:
                sub = rdf[rdf['tf'] == tf]
            if len(sub) == 0:
                continue
            best = sub.loc[sub['accuracy'].idxmax()]
            print(f"  {tf:>5s}:  period={int(best['period']):>3d}  "
                  f"sample={best['sample_min']:.1f}min  "
                  f"acc={best['accuracy']:.1f}%  "
                  f"MFE/MAE={best['ratio']:.1f}x  "
                  f"crosses={int(best['crosses'])}")

    # Save
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'reports/findings/dmi_sample_size_{ts_str}.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"DMI Sample Size Research -- {datetime.now()}\n")
        if results:
            pd.DataFrame(results).to_string(f)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()

"""
Validate SFE feed_bar() parity with batch_compute_states (ground truth).

For each TF bar close, compares:
  - BATCH: batch_compute_states(closed_bars_df)[-1]  (ground truth)
  - INCREMENTAL: feed_bar(bar)  (candidate)

Tests MarketState fields directly — no extract_79d involved.
Deposits raw 1s ticks in test dir, runs both paths, saves results.

Usage:
  python tools/validate_sfe_parity.py --setup --days 5   # copy test data
  python tools/validate_sfe_parity.py --days 2            # run parity check
  python tools/validate_sfe_parity.py --days 2 --verbose  # show every failure
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import TF_ORDER
from nn_v2.aggregator import Aggregator
from nn_v2.compute_79d import SFE_MIN_BARS, SFE_WINDOW

TEST_DIR = 'DATA/ATLAS_TEST'
ATLAS_1S_TEST = f'{TEST_DIR}/1s'
ATLAS_1S_PROD = 'DATA/ATLAS/1s'
TOLERANCE = 1e-6

# MarketState fields to compare (numeric only)
COMPARE_FIELDS = [
    'regression_center', 'regression_sigma', 'z_score', 'velocity',
    'P_at_center', 'P_near_upper', 'P_near_lower',
    'entropy', 'entropy_normalized',
    'hurst_exponent', 'adx_strength', 'dmi_plus', 'dmi_minus',
    'reversion_probability', 'breakout_probability', 'reversion_potential',
    'net_force', 'F_momentum', 'mean_reversion_force',
    'term_pid', 'oscillation_entropy_normalized', 'swing_noise_ticks',
    'momentum_strength',
]


def parse_args():
    p = argparse.ArgumentParser(description='Validate SFE feed_bar parity')
    p.add_argument('--days', type=int, default=2)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--setup', action='store_true', help='Copy test data from ATLAS')
    p.add_argument('--start', type=str, default=None)
    return p.parse_args()


def setup_test_data(n_days, start=None):
    import shutil
    files = sorted(glob.glob(os.path.join(ATLAS_1S_PROD, '*.parquet')))
    if start:
        start_key = start.replace('-', '_')
        files = [f for f in files if os.path.basename(f).replace('.parquet', '') >= start_key]
    if len(files) > n_days:
        step = len(files) // n_days
        files = [files[i * step] for i in range(n_days)]
    files = files[:n_days]
    os.makedirs(ATLAS_1S_TEST, exist_ok=True)
    for f in files:
        dst = os.path.join(ATLAS_1S_TEST, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
            print(f'  Copied: {os.path.basename(f)}')
    print(f'  Test data: {ATLAS_1S_TEST}/ ({len(files)} days)')


def get_day_files(n_days, start=None):
    files = sorted(glob.glob(os.path.join(ATLAS_1S_TEST, '*.parquet')))
    if not files:
        print(f'No test data in {ATLAS_1S_TEST}/')
        print(f'Run with --setup first.')
        return []
    if start:
        start_key = start.replace('-', '_')
        files = [f for f in files if os.path.basename(f).replace('.parquet', '') >= start_key]
    return files[:n_days]


def compare_states(batch_state, inc_state, tf, ts, verbose=False):
    """Compare two MarketState objects. Returns dict of failing fields."""
    failures = {}
    for field in COMPARE_FIELDS:
        b_val = float(getattr(batch_state, field, 0.0))
        i_val = float(getattr(inc_state, field, 0.0))
        diff = abs(b_val - i_val)
        if diff > TOLERANCE:
            failures[field] = (b_val, i_val, diff)
            if verbose:
                print(f'    {tf} ts={ts:.0f} {field}: batch={b_val:.8f} inc={i_val:.8f} diff={diff:.2e}')
    return failures


def run_one_day(day_file, verbose=False):
    """Run parity check on one day. Compares at MarketState level per TF bar close."""
    day_name = os.path.basename(day_file).replace('.parquet', '')
    df = pd.read_parquet(day_file).sort_values('timestamp').reset_index(drop=True)

    agg = Aggregator(history_limit=2000)
    sfe_batch = StatisticalFieldEngine()
    sfe_inc = {tf: StatisticalFieldEngine() for tf in TF_ORDER}

    pending_tf_bars = {}
    total_compared = 0
    total_failed = 0
    max_diffs = defaultdict(float)  # {field: max_diff}

    def on_bar_close(tf, bar):
        if tf in TF_ORDER:
            pending_tf_bars[tf] = bar

    agg.on_bar_close = on_bar_close

    for _, row in df.iterrows():
        bar = {
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row.get('volume', 0),
        }
        pending_tf_bars.clear()
        agg.feed(bar)

        for tf, tf_bar in pending_tf_bars.items():
            # INCREMENTAL: feed_bar
            inc_state = sfe_inc[tf].feed_bar(
                tf_bar['open'], tf_bar['high'], tf_bar['low'],
                tf_bar['close'], tf_bar['volume'], tf_bar['timestamp'])

            if inc_state is None:
                continue

            # BATCH: batch_compute_states on same bars from aggregator
            df_tf = agg.get_closed_bars_df(tf)
            if len(df_tf) < SFE_MIN_BARS:
                continue
            windowed = df_tf.tail(SFE_WINDOW).reset_index(drop=True) if len(df_tf) > SFE_WINDOW else df_tf
            batch_results = sfe_batch.batch_compute_states(windowed)
            if not batch_results:
                continue
            batch_state = batch_results[-1]['state']

            # COMPARE
            total_compared += 1
            failures = compare_states(batch_state, inc_state, tf, tf_bar['timestamp'], verbose)
            if failures:
                total_failed += 1
                for field, (_, _, diff) in failures.items():
                    if diff > max_diffs[field]:
                        max_diffs[field] = diff

    status = 'PASS' if total_failed == 0 else f'FAIL ({total_failed}/{total_compared})'
    overall_max = max(max_diffs.values()) if max_diffs else 0.0
    worst_field = max(max_diffs, key=max_diffs.get) if max_diffs else '-'

    return {
        'day': day_name,
        'compared': total_compared,
        'failed': total_failed,
        'status': status,
        'max_diffs': dict(max_diffs),
        'overall_max': overall_max,
        'worst_field': worst_field,
    }


def main():
    args = parse_args()

    if args.setup:
        print('Setting up test data...')
        setup_test_data(args.days, args.start)
        return

    day_files = get_day_files(args.days, args.start)
    if not day_files:
        return

    print(f'SFE PARITY VALIDATION -- {len(day_files)} days')
    print(f'  Comparing: batch_compute_states[-1] vs feed_bar()')
    print(f'  Fields: {len(COMPARE_FIELDS)} numeric MarketState fields')
    print(f'  Tolerance: {TOLERANCE}')
    print()

    all_results = []
    all_pass = True

    for day_file in tqdm(day_files, desc='Days', unit='day'):
        result = run_one_day(day_file, verbose=args.verbose)
        all_results.append(result)

        icon = 'OK' if 'PASS' in result['status'] else 'XX'
        tqdm.write(
            f'  {icon} {result["day"]}: {result["compared"]} comparisons | '
            f'max_diff={result["overall_max"]:.2e} | {result["status"]}')

        if 'FAIL' in result['status']:
            all_pass = False
            sorted_diffs = sorted(result['max_diffs'].items(), key=lambda x: -x[1])[:5]
            for fname, d in sorted_diffs:
                tqdm.write(f'      {fname}: {d:.2e}')

    total = sum(r['compared'] for r in all_results)
    worst = max(r['overall_max'] for r in all_results) if all_results else 0
    worst_f = max(all_results, key=lambda r: r['overall_max'])['worst_field'] if all_results else '?'

    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'  Days: {len(all_results)} | Comparisons: {total:,}')
    print(f'  Worst diff: {worst:.2e} ({worst_f})')
    print(f'  Verdict: {"PASS -- feed_bar matches batch" if all_pass else "FAIL"}')
    print(f'{"="*60}')

    # Save report
    os.makedirs('reports/findings', exist_ok=True)
    import time
    report_path = f'reports/findings/sfe_parity_{time.strftime("%Y-%m-%d")}.md'
    with open(report_path, 'w') as f:
        f.write(f'# SFE Parity Validation -- {time.strftime("%Y-%m-%d")}\n\n')
        f.write(f'Tolerance: {TOLERANCE}\n')
        f.write(f'Comparisons: {total:,} | Worst: {worst:.2e} ({worst_f})\n')
        f.write(f'Verdict: {"PASS" if all_pass else "FAIL"}\n\n')
        f.write('| Day | Comparisons | Max Diff | Worst | Status |\n')
        f.write('|-----|-------------|----------|-------|--------|\n')
        for r in all_results:
            f.write(f'| {r["day"]} | {r["compared"]} | {r["overall_max"]:.2e} | '
                    f'{r["worst_field"]} | {r["status"]} |\n')
    print(f'\nReport: {report_path}')


if __name__ == '__main__':
    main()

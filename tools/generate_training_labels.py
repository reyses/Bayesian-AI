"""
Training Label Generator — forward PnL at multiple hold durations.

For each 1m bar across all IS data:
  1. Compute 79D state (from closed bars only — zero lookahead on features)
  2. Look forward at multiple durations: 5, 15, 30, 60, 180 bars
  3. Record LONG and SHORT PnL + max drawdown + max favorable at each duration
  4. Determine best direction + duration (risk-adjusted)
  5. Save: 79D features + labels as parquet

Output: DATA/TRAINING_LABELS/YYYY_MM_DD.parquet
  Columns: 79D features + label columns

Usage:
  python tools/generate_training_labels.py                    # all IS days
  python tools/generate_training_labels.py --start 2025-06-01 # from date
  python tools/generate_training_labels.py --days 10          # first N days
  python tools/generate_training_labels.py --end 2025-12-31   # up to date

Spec: docs/Active/FEATURE_VECTOR_79D_SPEC.md
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import gc
import glob
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features_79d import (
    extract_79d, build_all_tf_ohlcv, FEATURE_NAMES_79D, N_FEATURES,
    TF_ORDER, TF_SECONDS
)
from core.statistical_field_engine import StatisticalFieldEngine

TICK = 0.25
TV = 0.50

# Hold durations to evaluate (in 1m bars)
# Finer resolution where R/R changes fastest (1-20 bars), coarser after
HOLD_DURATIONS = [1, 3, 5, 10, 15, 20, 30, 60]

# Risk penalty for drawdown (risk_adj_pnl = pnl - RISK_PENALTY * max_drawdown)
RISK_PENALTY = 0.5

# Minimum edge required to label as a trade (otherwise label = no_trade)
MIN_EDGE_DOLLARS = 1.0  # must exceed cost + buffer

# Paths
ATLAS_1M = 'DATA/ATLAS/1m'
OUTPUT_DIR = 'DATA/TRAINING_LABELS'

# History days to load for higher TF context (SFE needs 21+ 1h bars ≈ 21 days)
HISTORY_DAYS = 25


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Generate training labels for 79D NN')
    p.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD')
    p.add_argument('--days', type=int, default=None, help='Process first N days only')
    p.add_argument('--oos-start', type=str, default='2026-01-01',
                   help='OOS boundary — only generate labels for IS (before this date)')
    return p.parse_args()


def compute_forward_labels(closes: np.ndarray, bar_idx: int, durations: list) -> dict:
    """Compute forward PnL + risk profile for LONG and SHORT at multiple durations.

    For each duration, records:
      - pnl: final PnL at bar N
      - mdd: max drawdown on the path (negative = adverse)
      - mfe: max favorable excursion (peak profit before exit)
      - mdd_bar: which bar the max drawdown occurred (how fast the pain came)
      - recovery: did price recover from MDD to profitable? (1=yes, 0=no)
      - reward_risk: pnl / abs(mdd) — risk-adjusted return ratio
      - path_efficiency: pnl / mfe — how much of the peak was captured at exit

    Args:
        closes: full array of 1m close prices for the day
        bar_idx: current bar index (entry point)
        durations: list of hold durations in bars

    Returns:
        dict with per-duration labels + best selection + risk summary
    """
    entry_price = closes[bar_idx]
    n = len(closes)
    labels = {}

    nan_cols = ['pnl', 'mdd', 'mfe', 'mdd_bar', 'recovery', 'reward_risk', 'path_efficiency']

    for dur in durations:
        end_idx = bar_idx + dur
        if end_idx >= n:
            for prefix in nan_cols:
                for side in ['long', 'short']:
                    labels[f'{prefix}_{side}_{dur}'] = np.nan
            continue

        fwd_closes = closes[bar_idx + 1:end_idx + 1]
        if len(fwd_closes) == 0:
            for prefix in nan_cols:
                for side in ['long', 'short']:
                    labels[f'{prefix}_{side}_{dur}'] = np.nan
            continue

        for side in ['long', 'short']:
            if side == 'long':
                pnl = (fwd_closes[-1] - entry_price) / TICK * TV
                path = (fwd_closes - entry_price) / TICK * TV
            else:
                pnl = (entry_price - fwd_closes[-1]) / TICK * TV
                path = (entry_price - fwd_closes) / TICK * TV

            mfe = np.max(path)
            mdd = np.min(path)
            mdd_bar = int(np.argmin(path)) + 1  # 1-indexed: which bar was the worst

            # Did price recover from worst drawdown to profit?
            if mdd < 0:
                # After the MDD bar, did we get back to positive?
                post_mdd_path = path[np.argmin(path):]
                recovery = 1.0 if np.any(post_mdd_path > 0) else 0.0
            else:
                recovery = 1.0  # never went negative

            # Reward/risk ratio
            abs_mdd = abs(min(0, mdd))
            reward_risk = pnl / abs_mdd if abs_mdd > 0.01 else (999.0 if pnl > 0 else 0.0)

            # Path efficiency: how much of the peak was captured?
            path_efficiency = pnl / mfe if mfe > 0.01 else 0.0

            labels[f'pnl_{side}_{dur}'] = pnl
            labels[f'mdd_{side}_{dur}'] = mdd
            labels[f'mfe_{side}_{dur}'] = mfe
            labels[f'mdd_bar_{side}_{dur}'] = mdd_bar
            labels[f'recovery_{side}_{dur}'] = recovery
            labels[f'reward_risk_{side}_{dur}'] = reward_risk
            labels[f'path_efficiency_{side}_{dur}'] = path_efficiency

    # === DIRECTION CLARITY: does direction hold across durations? ===
    # If LONG wins at bar 1, 3, 5, 10, 15 → high clarity (consistent direction)
    # If LONG wins at bar 1 but SHORT wins at bar 5 → low clarity (direction flips)
    # Low clarity = the market is choppy, default to shortest hold

    # Step 1: determine winning direction at each duration
    dir_per_dur = {}
    for dur in durations:
        long_pnl = labels.get(f'pnl_long_{dur}', np.nan)
        short_pnl = labels.get(f'pnl_short_{dur}', np.nan)
        if np.isnan(long_pnl) or np.isnan(short_pnl):
            dir_per_dur[dur] = 'none'
        elif long_pnl > short_pnl:
            dir_per_dur[dur] = 'long'
        elif short_pnl > long_pnl:
            dir_per_dur[dur] = 'short'
        else:
            dir_per_dur[dur] = 'none'

    # Step 2: for each duration, clarity = what fraction of durations UP TO this one agree
    for i, dur in enumerate(durations):
        dirs_so_far = [dir_per_dur[d] for d in durations[:i + 1] if dir_per_dur[d] != 'none']
        if not dirs_so_far:
            labels[f'dir_clarity_{dur}'] = 0.0
            continue
        # Count how many agree with the majority
        long_count = sum(1 for d in dirs_so_far if d == 'long')
        short_count = sum(1 for d in dirs_so_far if d == 'short')
        majority = max(long_count, short_count)
        labels[f'dir_clarity_{dur}'] = majority / len(dirs_so_far)

    # Step 3: overall direction consistency across ALL durations
    all_dirs = [dir_per_dur[d] for d in durations if dir_per_dur[d] != 'none']
    if all_dirs:
        long_count = sum(1 for d in all_dirs if d == 'long')
        short_count = sum(1 for d in all_dirs if d == 'short')
        labels['direction_consistency'] = max(long_count, short_count) / len(all_dirs)
    else:
        labels['direction_consistency'] = 0.0

    # === BEST SELECTION: conservative — default to shortest safe hold ===
    # Logic: start from 1-bar hold (safest). Only extend to longer duration
    # if the longer hold is BOTH more profitable AND directionally clear.
    # This prevents the NN from learning to hold through ambiguity.

    best_dir = 'none'
    best_dur = 0
    best_pnl = 0.0
    best_mdd = 0.0
    best_mfe = 0.0
    best_reward_risk = 0.0
    best_risk_adj = -999.0
    best_clarity = 0.0

    # First pass: find the best option at each duration
    candidates = []
    for dur in durations:
        clarity = labels.get(f'dir_clarity_{dur}', 0.0)

        for side in ['long', 'short']:
            pnl = labels.get(f'pnl_{side}_{dur}', np.nan)
            mdd = labels.get(f'mdd_{side}_{dur}', np.nan)
            mfe = labels.get(f'mfe_{side}_{dur}', np.nan)
            rr = labels.get(f'reward_risk_{side}_{dur}', 0.0)

            if np.isnan(pnl) or np.isnan(mdd):
                continue
            if pnl <= MIN_EDGE_DOLLARS:
                continue

            drawdown = abs(min(0, mdd))
            risk_adj = pnl - RISK_PENALTY * drawdown

            # Direction clarity gate: longer holds REQUIRE higher clarity
            # 1-bar hold: any clarity OK (0.5 minimum)
            # 60-bar hold: needs near-perfect clarity
            # Formula: required_clarity = 0.5 + 0.5 * (dur / max_dur)
            max_dur = max(durations)
            required_clarity = 0.5 + 0.5 * (dur / max_dur)
            clarity_surplus = max(0, clarity - required_clarity)

            # Score: risk-adjusted PnL weighted by clarity surplus
            # If clarity < required → surplus = 0 → this duration scores 0
            # If clarity = 1.0 → full credit
            clarity_adj = risk_adj * clarity_surplus / (1.0 - required_clarity + 0.01)

            candidates.append({
                'side': side, 'dur': dur, 'pnl': pnl, 'mdd': mdd,
                'mfe': mfe if not np.isnan(mfe) else 0.0,
                'rr': rr if not np.isnan(rr) else 0.0,
                'risk_adj': risk_adj, 'clarity': clarity,
                'score': clarity_adj,
            })

    if candidates:
        # Sort by score — clarity-adjusted risk-adjusted PnL
        candidates.sort(key=lambda c: c['score'], reverse=True)
        winner = candidates[0]

        best_dir = winner['side']
        best_dur = winner['dur']
        best_pnl = winner['pnl']
        best_mdd = winner['mdd']
        best_mfe = winner['mfe']
        best_reward_risk = winner['rr']
        best_risk_adj = winner['risk_adj']
        best_clarity = winner['clarity']

    labels['best_direction'] = best_dir
    labels['best_duration'] = best_dur
    labels['best_pnl'] = best_pnl
    labels['best_risk_adj'] = best_risk_adj if best_risk_adj > -999 else 0.0
    labels['best_mdd'] = best_mdd
    labels['best_mfe'] = best_mfe
    labels['best_reward_risk'] = best_reward_risk
    labels['best_clarity'] = best_clarity

    # === RISK SUMMARY: what the execution layer needs ===
    # Expected drawdown for the chosen duration (even if best = none)
    if best_dir != 'none' and best_dur > 0:
        labels['expected_drawdown'] = abs(min(0, best_mdd))
        labels['expected_peak'] = best_mfe
        labels['drawdown_bar'] = labels.get(f'mdd_bar_{best_dir}_{best_dur}', 0)
        labels['recovery_rate'] = labels.get(f'recovery_{best_dir}_{best_dur}', 0.0)
        labels['path_efficiency'] = labels.get(f'path_efficiency_{best_dir}_{best_dur}', 0.0)
    else:
        labels['expected_drawdown'] = 0.0
        labels['expected_peak'] = 0.0
        labels['drawdown_bar'] = 0
        labels['recovery_rate'] = 0.0
        labels['path_efficiency'] = 0.0

    return labels


def process_day(day_file: str, history_1m: pd.DataFrame) -> pd.DataFrame:
    """Process one day: compute 79D features + forward labels for every bar.

    Args:
        day_file: path to today's 1m parquet
        history_1m: previous days' 1m bars (for higher TF context)

    Returns:
        DataFrame with 79D features + label columns. One row per bar.
    """
    today_1m = pd.read_parquet(day_file).sort_values('timestamp').reset_index(drop=True)
    if len(today_1m) < 50:  # skip sparse days
        return pd.DataFrame()

    closes = today_1m['close'].values
    n_bars = len(today_1m)

    # Build higher TF OHLCV from closed bars (with history for 1h/1D)
    # We'll compute features incrementally: at bar i, use bars 0..i-1 as closed
    sfe = StatisticalFieldEngine()
    rows = []
    prev_velocities = {}

    # Pre-compute SFE for the full day on 1m (we'll use states up to current bar)
    all_ohlcv_full = build_all_tf_ohlcv(today_1m, historical_1m_bars=history_1m)

    # Run SFE once per TF on the full data (training mode — labels use future anyway)
    states_by_tf_all = {}
    for tf in TF_ORDER:
        if tf in all_ohlcv_full and len(all_ohlcv_full[tf]) >= 21:
            states = sfe.batch_compute_states(all_ohlcv_full[tf])
            if states:
                states_by_tf_all[tf] = states

    del sfe
    gc.collect()

    # For each 1m bar, look up the corresponding state at each TF
    # Align by timestamp
    tf_timestamps = {}
    for tf in TF_ORDER:
        if tf in all_ohlcv_full and 'timestamp' in all_ohlcv_full[tf].columns:
            tf_timestamps[tf] = all_ohlcv_full[tf]['timestamp'].values

    today_ts = today_1m['timestamp'].values

    # Find the index range for today's bars within the full 1m (which includes history)
    full_1m_ts = all_ohlcv_full['1m']['timestamp'].values
    today_start_idx = np.searchsorted(full_1m_ts, today_ts[0], side='left')

    for bar_idx in range(n_bars):
        ts = today_ts[bar_idx]

        # Build states_by_tf for this bar
        states_this_bar = {}
        for tf in TF_ORDER:
            if tf not in states_by_tf_all:
                continue
            tf_states = states_by_tf_all[tf]

            if tf == '1m':
                # Direct index: offset by history length
                state_idx = today_start_idx + bar_idx
                if 0 <= state_idx < len(tf_states):
                    states_this_bar[tf] = tf_states[state_idx]
            else:
                # Align by timestamp: latest TF bar <= current timestamp
                if tf in tf_timestamps:
                    tf_ts = tf_timestamps[tf]
                    tf_bar_idx = np.searchsorted(tf_ts, ts, side='right') - 1
                    if 0 <= tf_bar_idx < len(tf_states):
                        states_this_bar[tf] = tf_states[tf_bar_idx]

        # Extract 79D
        feat, prev_velocities = extract_79d(
            states_this_bar, all_ohlcv_full, prev_velocities, ts
        )

        # Compute forward labels
        labels = compute_forward_labels(closes, bar_idx, HOLD_DURATIONS)

        # Build row
        row = {'timestamp': ts}
        for fi, fname in enumerate(FEATURE_NAMES_79D):
            row[fname] = feat[fi]
        row.update(labels)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all 1m files
    all_files = sorted(glob.glob(os.path.join(ATLAS_1M, '*.parquet')))
    print(f'Total 1m files: {len(all_files)}')

    # Filter by date range
    def file_date(f):
        return os.path.basename(f).replace('.parquet', '').replace('_', '-')

    if args.start:
        all_files = [f for f in all_files if file_date(f) >= args.start]
    if args.end:
        all_files = [f for f in all_files if file_date(f) <= args.end]
    if args.oos_start:
        all_files = [f for f in all_files if file_date(f) < args.oos_start]
    if args.days:
        all_files = all_files[:args.days]

    print(f'Processing: {len(all_files)} days')
    if all_files:
        print(f'  From: {file_date(all_files[0])}')
        print(f'  To:   {file_date(all_files[-1])}')

    # Process each day
    total_bars = 0
    total_trades = 0

    for file_idx, day_file in enumerate(tqdm(all_files, desc='Days', unit='day')):
        day_name = os.path.basename(day_file).replace('.parquet', '')
        out_path = os.path.join(OUTPUT_DIR, f'{day_name}.parquet')

        # Skip if already generated
        if os.path.exists(out_path):
            continue

        # Load history: previous HISTORY_DAYS of 1m data
        history_start = max(0, file_idx - HISTORY_DAYS)
        history_files = all_files[history_start:file_idx]
        if history_files:
            history_dfs = [pd.read_parquet(f) for f in history_files]
            history_1m = pd.concat(history_dfs, ignore_index=True).sort_values('timestamp')
            del history_dfs
        else:
            history_1m = pd.DataFrame()

        # Process the day
        df = process_day(day_file, history_1m)
        del history_1m
        gc.collect()

        if len(df) == 0:
            continue

        # Save
        df.to_parquet(out_path, index=False)

        # Stats
        n_bars = len(df)
        n_trades = (df['best_direction'] != 'none').sum()
        total_bars += n_bars
        total_trades += n_trades

        # Update progress bar
        if (file_idx + 1) % 20 == 0:
            tqdm.write(f'  {day_name}: {n_bars} bars, {n_trades} tradeable '
                      f'| cumul: {total_bars:,} bars, {total_trades:,} trades')

    print(f'\nDone: {total_bars:,} bars, {total_trades:,} tradeable labels')
    print(f'Output: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()

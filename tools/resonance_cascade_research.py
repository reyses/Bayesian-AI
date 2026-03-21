"""Resonance Cascade Research — Multi-TF Peak Agreement Detection.

Hypothesis: When peak detection fires on ALL TF pairs simultaneously
in the same direction, that's a crash/rally. When <3 pairs agree, it's chop.

Each TF pair: child detects peak, parent validates (real/fake).
A "trend" is the decay of peaks in one direction over time.

Usage: python tools/resonance_cascade_research.py [--data DATA/ATLAS_OOS]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine


# TF pairs: child -> parent (child detects, parent validates)
TF_PAIRS = [
    ('15s', '1m'),
    ('1m',  '5m'),
    ('5m',  '15m'),
    ('15m', '1h'),
    ('1h',  '4h'),
]

TF_SECONDS = {
    '1s': 1, '5s': 5, '15s': 15, '30s': 30, '1m': 60,
    '2m': 120, '3m': 180, '5m': 300, '15m': 900,
    '30m': 1800, '1h': 3600, '4h': 14400, '1D': 86400,
}


def load_tf_states(data_root, tf_label):
    """Load parquets and compute states for a TF."""
    tf_dir = os.path.join(data_root, tf_label)
    if not os.path.isdir(tf_dir):
        return None, None

    files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
    if not files:
        return None, None

    chunks = []
    for fn in files:
        df = pd.read_parquet(os.path.join(tf_dir, fn))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        chunks.append(df)

    df = pd.concat(chunks, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

    engine = StatisticalFieldEngine()
    states = engine.batch_compute_states(df, use_cuda=True)

    timestamps = df['timestamp'].values.astype(np.float64)
    return timestamps, states


def detect_peaks(states):
    """Run peak detection on a sequence of states. Returns list of (index, direction)."""
    peaks = []
    prev_pc = 0.0
    prev_fm = 0.0

    for i, s in enumerate(states):
        ms = s['state'] if isinstance(s, dict) and 'state' in s else s
        pc = getattr(ms, 'P_at_center', 0.0) or 0.0
        fm = abs(getattr(ms, 'F_momentum', 0.0) or 0.0)
        coh = getattr(ms, 'oscillation_entropy_normalized', 0.0) or 0.0

        if i > 0 and prev_pc > 0.01:
            pc_delta = (pc - prev_pc) / max(abs(prev_pc), 1e-6)
            fm_delta = (fm - prev_fm) / max(abs(prev_fm), 1e-6) if prev_fm > 0.5 else 0.0

            pc_up = pc_delta > 0.05
            fm_down = fm_delta < -0.10

            if (pc_up or fm_down) and coh > 0.55:
                # Direction: if momentum was positive and decaying, peak is SHORT
                raw_fm = getattr(ms, 'F_momentum', 0.0) or 0.0
                direction = 'SHORT' if raw_fm > 0 else 'LONG'
                peaks.append((i, direction))

        prev_pc = pc
        prev_fm = fm

    return peaks


def compute_peak_agreement(all_peaks, all_timestamps, window_seconds=300):
    """At each timestamp, count how many TFs have a recent peak in the same direction.

    Returns DataFrame with: timestamp, n_long_peaks, n_short_peaks, max_agreement,
    cascade_direction, is_cascade (4+ agree).
    """
    # Build a unified timeline from the 1m TF (good resolution, not too granular)
    if '1m' not in all_timestamps:
        # Fallback to first available
        ref_tf = list(all_timestamps.keys())[0]
    else:
        ref_tf = '1m'

    ref_ts = all_timestamps[ref_tf]
    results = []

    for i, ts in enumerate(ref_ts):
        n_long = 0
        n_short = 0
        tf_details = {}

        for tf_label, peaks in all_peaks.items():
            tf_ts = all_timestamps[tf_label]

            # Find most recent peak within window
            recent_peak = None
            for peak_idx, peak_dir in reversed(peaks):
                if peak_idx < len(tf_ts):
                    peak_ts = tf_ts[peak_idx]
                    if ts - peak_ts <= window_seconds and ts - peak_ts >= 0:
                        recent_peak = peak_dir
                        break

            if recent_peak == 'LONG':
                n_long += 1
                tf_details[tf_label] = 'LONG'
            elif recent_peak == 'SHORT':
                n_short += 1
                tf_details[tf_label] = 'SHORT'

        max_agree = max(n_long, n_short)
        cascade_dir = 'LONG' if n_long > n_short else ('SHORT' if n_short > n_long else 'NEUTRAL')
        is_cascade = max_agree >= 4

        results.append({
            'timestamp': ts,
            'n_long': n_long,
            'n_short': n_short,
            'max_agreement': max_agree,
            'cascade_direction': cascade_dir,
            'is_cascade': is_cascade,
            'details': str(tf_details),
        })

    return pd.DataFrame(results)


def evaluate_cascade_accuracy(agreement_df, price_ts, price_close, lookahead_bars=20):
    """Check if cascade direction predicts future price movement."""
    results = []

    for _, row in agreement_df.iterrows():
        if row['max_agreement'] < 2:
            continue

        ts = row['timestamp']
        idx = int(np.searchsorted(price_ts, ts, side='right')) - 1
        if idx < 0 or idx + lookahead_bars >= len(price_close):
            continue

        entry_price = price_close[idx]
        future_prices = price_close[idx + 1: idx + lookahead_bars + 1]
        if len(future_prices) == 0:
            continue

        if row['cascade_direction'] == 'LONG':
            mfe = (max(future_prices) - entry_price) / 0.25  # ticks
            mae = (entry_price - min(future_prices)) / 0.25
            final = (future_prices[-1] - entry_price) / 0.25
        elif row['cascade_direction'] == 'SHORT':
            mfe = (entry_price - min(future_prices)) / 0.25
            mae = (max(future_prices) - entry_price) / 0.25
            final = (entry_price - future_prices[-1]) / 0.25
        else:
            continue

        results.append({
            'timestamp': ts,
            'agreement': row['max_agreement'],
            'direction': row['cascade_direction'],
            'is_cascade': row['is_cascade'],
            'mfe_ticks': mfe,
            'mae_ticks': mae,
            'final_ticks': final,
            'profitable': final > 0,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Resonance Cascade Research')
    parser.add_argument('--data', default='DATA/ATLAS_OOS', help='Data root')
    parser.add_argument('--window', type=int, default=300, help='Peak agreement window (seconds)')
    parser.add_argument('--lookahead', type=int, default=20, help='Lookahead bars for accuracy check')
    args = parser.parse_args()

    print('=' * 70)
    print('RESONANCE CASCADE RESEARCH')
    print(f'Data: {args.data}  |  Window: {args.window}s  |  Lookahead: {args.lookahead} bars')
    print('=' * 70)

    # Load states for each TF in the pairs
    tfs_needed = set()
    for child, parent in TF_PAIRS:
        tfs_needed.add(child)
        tfs_needed.add(parent)

    all_timestamps = {}
    all_states = {}
    all_peaks = {}

    for tf in sorted(tfs_needed, key=lambda x: TF_SECONDS.get(x, 0)):
        print(f'  Loading {tf}...', end=' ', flush=True)
        ts, states = load_tf_states(args.data, tf)
        if ts is None:
            print('MISSING')
            continue
        all_timestamps[tf] = ts
        all_states[tf] = states
        peaks = detect_peaks(states)
        all_peaks[tf] = peaks
        n_long = sum(1 for _, d in peaks if d == 'LONG')
        n_short = sum(1 for _, d in peaks if d == 'SHORT')
        print(f'{len(states):,} states, {len(peaks):,} peaks ({n_long}L / {n_short}S)')

    if len(all_peaks) < 3:
        print('ERROR: Need at least 3 TFs with data')
        return

    # Compute agreement
    print(f'\nComputing peak agreement (window={args.window}s)...')
    agreement = compute_peak_agreement(all_peaks, all_timestamps, args.window)
    print(f'  Total reference bars: {len(agreement):,}')

    # Agreement distribution
    print(f'\n=== AGREEMENT DISTRIBUTION ===')
    for n in range(6):
        count = (agreement['max_agreement'] == n).sum()
        pct = count / len(agreement) * 100
        label = '  CASCADE!' if n >= 4 else ''
        print(f'  {n} TFs agree: {count:>6,} bars ({pct:>5.1f}%){label}')

    cascade_bars = agreement[agreement['is_cascade']].copy()
    print(f'\n  Cascade events (4+ agree): {len(cascade_bars):,} bars')
    if len(cascade_bars) > 0:
        cascade_bars['dt'] = pd.to_datetime(cascade_bars['timestamp'], unit='s')
        print(f'  Direction: LONG={sum(cascade_bars["cascade_direction"]=="LONG")}, '
              f'SHORT={sum(cascade_bars["cascade_direction"]=="SHORT")}')

    # Check accuracy: does agreement predict future direction?
    print(f'\n=== ACCURACY CHECK (lookahead={args.lookahead} bars) ===')
    # Use 1m price for accuracy check
    if '1m' in all_timestamps:
        _1m_ts = all_timestamps['1m']
        _1m_close = np.array([
            getattr(s['state'] if isinstance(s, dict) else s, 'price', 0.0)
            for s in all_states['1m']
        ])
        accuracy = evaluate_cascade_accuracy(agreement, _1m_ts, _1m_close, args.lookahead)

        if len(accuracy) > 0:
            print(f'  Total signals evaluated: {len(accuracy):,}')
            print(f'\n  By agreement level:')
            print(f'  {"Agree":>6} {"N":>6} {"WR":>6} {"Avg MFE":>9} {"Avg MAE":>9} {"Avg Final":>10} {"PF":>6}')
            for n in sorted(accuracy['agreement'].unique()):
                sub = accuracy[accuracy['agreement'] == n]
                wr = sub['profitable'].mean() * 100
                avg_mfe = sub['mfe_ticks'].mean()
                avg_mae = sub['mae_ticks'].mean()
                avg_final = sub['final_ticks'].mean()
                gw = sub[sub['final_ticks'] > 0]['final_ticks'].sum()
                gl = abs(sub[sub['final_ticks'] < 0]['final_ticks'].sum())
                pf = gw / gl if gl > 0 else 0
                label = ' <-- CASCADE' if n >= 4 else ''
                print(f'  {n:>6} {len(sub):>6} {wr:>5.1f}% {avg_mfe:>8.1f}t {avg_mae:>8.1f}t {avg_final:>+9.1f}t {pf:>5.2f}{label}')

            # Feb 9 specific
            accuracy['dt'] = pd.to_datetime(accuracy['timestamp'], unit='s')
            feb9 = accuracy[accuracy['dt'].dt.date == pd.Timestamp('2026-02-09').date()]
            if len(feb9) > 0:
                print(f'\n=== FEB 9 CASCADE SIGNALS ===')
                print(f'  Signals: {len(feb9)}, Cascades (4+): {(feb9["agreement"]>=4).sum()}')
                print(f'  Direction: LONG={sum(feb9["direction"]=="LONG")}, SHORT={sum(feb9["direction"]=="SHORT")}')
                print(f'  Avg MFE: {feb9["mfe_ticks"].mean():.1f}t, Avg Final: {feb9["final_ticks"].mean():+.1f}t')
                print(f'  WR: {feb9["profitable"].mean()*100:.1f}%')
        else:
            print('  No signals to evaluate')
    else:
        print('  1m data not available for accuracy check')

    # Save results
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'resonance_cascade_results.txt')
    # Redirect summary to file
    import io
    buf = io.StringIO()
    agreement.to_csv(os.path.join(out_dir, 'resonance_cascade_agreement.csv'), index=False)
    if 'accuracy' in dir() and len(accuracy) > 0:
        accuracy.to_csv(os.path.join(out_dir, 'resonance_cascade_accuracy.csv'), index=False)
    print(f'\n  Results saved to {out_dir}/')
    print('=' * 70)


if __name__ == '__main__':
    main()

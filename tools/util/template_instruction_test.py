"""Template Instruction Test — Apply IS templates to OOS, follow their instructions.

When a template matches the current bar's features:
  - Enter in the template's direction (long_bias vs short_bias)
  - Hold for the template's duration (avg_mfe_bar)
  - Exit at that bar
  - No gates, no exit cascade, pure template instruction following

This tests: do IS-learned templates give valid trade instructions on unseen OOS data?

Usage:
    python tools/template_instruction_test.py
    python tools/template_instruction_test.py --data DATA/ATLAS_OOS
    python tools/template_instruction_test.py --match-threshold 5.0
"""

import argparse
import csv
import glob
import os
import pickle
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.statistical_field_engine import StatisticalFieldEngine
from core_v2.feature_extraction import extract_feature_vector


TICK_SIZE = 0.25
TICK_VALUE = 0.50  # MNQ


def load_checkpoints(checkpoint_dir='checkpoints'):
    """Load pattern library and scaler from IS checkpoints."""
    lib_path = os.path.join(checkpoint_dir, 'pattern_library.pkl')
    scaler_path = os.path.join(checkpoint_dir, 'clustering_scaler.pkl')

    with open(lib_path, 'rb') as f:
        library = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Build centroid matrix
    template_ids = []
    centroids = []
    for tid, tmpl in library.items():
        if 'centroid' in tmpl and tmpl['centroid'] is not None:
            template_ids.append(tid)
            centroids.append(tmpl['centroid'])

    centroids = np.array(centroids)
    n_expected = scaler.n_features_in_
    # Pad centroids if needed
    if centroids.shape[1] < n_expected:
        pad = np.zeros((centroids.shape[0], n_expected - centroids.shape[1]))
        centroids = np.hstack([centroids, pad])

    # Pre-scale centroids
    centroids_scaled = scaler.transform(centroids)

    print(f'Loaded {len(library)} templates, {len(centroids)} matchable')
    print(f'Scaler expects {n_expected}D features')
    return library, scaler, template_ids, centroids_scaled


def extract_features_from_state(state, tf_seconds=15, depth=8):
    """Extract 16D feature vector from MarketState (same as AdvanceEngine._build_features)."""
    feat = extract_feature_vector(
        z_score=getattr(state, 'z_score', 0.0),
        velocity=getattr(state, 'velocity', 0.0),
        momentum=getattr(state, 'momentum_strength',
                         getattr(state, 'momentum', 0.0)),
        entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
        tf_seconds=tf_seconds,
        depth=float(depth),
        parent_is_band_reversal=0.0,
        adx=getattr(state, 'adx_strength', 0.0) / 100.0,
        hurst=getattr(state, 'hurst_exponent', 0.5),
        dmi_diff=(getattr(state, 'dmi_plus', 0.0)
                  - getattr(state, 'dmi_minus', 0.0)) / 100.0,
        parent_z=0.0,
        parent_dmi_diff=0.0,
        root_is_roche=0.0,
        tf_alignment=0.0,
        pid=getattr(state, 'term_pid', 0.0),
        osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0),
    )
    return feat


def match_template(feat, scaler, centroids_scaled, template_ids, library,
                   threshold=3.0):
    """Match feature vector to nearest template. Returns (tid, dist, lib_entry) or None."""
    n_expected = scaler.n_features_in_
    feat_arr = np.array(feat).reshape(1, -1)
    if feat_arr.shape[1] < n_expected:
        feat_arr = np.pad(feat_arr, ((0, 0), (0, n_expected - feat_arr.shape[1])))
    elif feat_arr.shape[1] > n_expected:
        feat_arr = feat_arr[:, :n_expected]

    feat_scaled = scaler.transform(feat_arr)
    dists = np.linalg.norm(centroids_scaled - feat_scaled, axis=1)
    best_idx = np.argmin(dists)
    best_dist = dists[best_idx]

    if best_dist > threshold:
        return None

    tid = template_ids[best_idx]
    return tid, best_dist, library[tid]


def get_template_instruction(lib_entry):
    """Extract direction and hold duration from template."""
    long_bias = lib_entry.get('long_bias', 0.5)
    short_bias = lib_entry.get('short_bias', 0.5)
    direction = 'LONG' if long_bias > short_bias else 'SHORT'

    # Hold duration in 1m bars, convert to 15s bars (x4)
    hold_1m = lib_entry.get('avg_mfe_bar', 11)
    if hold_1m is None or hold_1m <= 0:
        hold_1m = 11
    hold_15s = int(hold_1m * 4)  # 1m bars to 15s bars
    hold_15s = max(4, min(hold_15s, 120))  # clamp 1-30 min

    return direction, hold_15s


def run_test(data_path, checkpoint_dir='checkpoints', match_threshold=3.0):
    """Run the template instruction test on OOS data."""

    # Load templates
    library, scaler, template_ids, centroids_scaled = load_checkpoints(checkpoint_dir)

    # Load OOS 15s data
    tf_dir = os.path.join(data_path, '15s')
    files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    if not files:
        print(f'No 15s parquet files in {tf_dir}')
        return

    print(f'Loading {len(files)} OOS files...')
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f'OOS bars: {len(df)}')

    # Compute states
    print('Computing states (CUDA)...')
    engine = StatisticalFieldEngine()
    states = engine.batch_compute_states(df)
    print(f'States: {len(states)}')

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df['timestamp'].values

    # Simulate
    trades = []
    in_trade = False
    trade_entry_price = 0.0
    trade_dir = ''
    trade_exit_bar = 0
    trade_entry_bar = 0
    trade_tid = None

    matches = 0
    no_match = 0

    print(f'Simulating (threshold={match_threshold})...')
    for i in tqdm(range(len(states)), desc='OOS bars', miniters=1000):
        # Exit check
        if in_trade and i >= trade_exit_bar:
            exit_price = closes[i]
            if trade_dir == 'LONG':
                pnl_ticks = (exit_price - trade_entry_price) / TICK_SIZE
                mae = (trade_entry_price - min(lows[trade_entry_bar:i + 1])) / TICK_SIZE
                mfe = (max(highs[trade_entry_bar:i + 1]) - trade_entry_price) / TICK_SIZE
            else:
                pnl_ticks = (trade_entry_price - exit_price) / TICK_SIZE
                mae = (max(highs[trade_entry_bar:i + 1]) - trade_entry_price) / TICK_SIZE
                mfe = (trade_entry_price - min(lows[trade_entry_bar:i + 1])) / TICK_SIZE

            trades.append({
                'entry_bar': trade_entry_bar,
                'exit_bar': i,
                'entry_price': trade_entry_price,
                'exit_price': exit_price,
                'direction': trade_dir,
                'pnl_ticks': pnl_ticks,
                'pnl_dollars': pnl_ticks * TICK_VALUE,
                'mae_ticks': mae,
                'mfe_ticks': mfe,
                'hold_bars': i - trade_entry_bar,
                'template_id': trade_tid,
                'timestamp': timestamps[trade_entry_bar],
            })
            in_trade = False

        # Entry check (only when flat)
        if not in_trade:
            state = states[i]
            if state is None:
                no_match += 1
                continue

            feat = extract_features_from_state(state, tf_seconds=15, depth=8)
            result = match_template(feat, scaler, centroids_scaled,
                                    template_ids, library, match_threshold)
            if result is None:
                no_match += 1
                continue

            tid, dist, lib_entry = result
            direction, hold_bars = get_template_instruction(lib_entry)

            trade_entry_price = closes[i]
            trade_dir = direction
            trade_exit_bar = i + hold_bars
            trade_entry_bar = i
            trade_tid = tid
            in_trade = True
            matches += 1

    # Results
    df_trades = pd.DataFrame(trades)
    if len(df_trades) == 0:
        print('No trades generated')
        return

    wins = df_trades[df_trades.pnl_ticks > 0]
    losses = df_trades[df_trades.pnl_ticks <= 0]
    total_pnl = df_trades.pnl_dollars.sum()
    n_days = len(df) / 2400  # approx 15s bars per session day

    # Report
    report = []
    report.append('=' * 70)
    report.append('TEMPLATE INSTRUCTION TEST — IS Templates on OOS')
    report.append(f'Match threshold: {match_threshold}')
    report.append('Rule: template match -> enter direction -> hold duration -> exit')
    report.append('No gates. No exit cascade. Pure template instructions.')
    report.append('=' * 70)
    report.append('')
    report.append(f'OOS bars: {len(df):,}')
    report.append(f'Matches: {matches:,} | No match: {no_match:,}')
    report.append(f'Trades: {len(df_trades)}')
    report.append(f'WR: {len(wins) / len(df_trades) * 100:.1f}%')
    report.append(f'Total PnL: {df_trades.pnl_ticks.sum():.0f}t (${total_pnl:,.1f})')
    report.append(f'Avg PnL: {df_trades.pnl_ticks.mean():.1f}t (${df_trades.pnl_dollars.mean():.2f})')
    report.append(f'Avg MFE: {df_trades.mfe_ticks.mean():.1f}t | Avg MAE: {df_trades.mae_ticks.mean():.1f}t')
    report.append(f'MFE/MAE: {df_trades.mfe_ticks.mean() / max(df_trades.mae_ticks.mean(), 1):.1f}x')
    report.append(f'Avg hold: {df_trades.hold_bars.mean():.0f} bars ({df_trades.hold_bars.mean() * 15 / 60:.1f} min)')
    report.append(f'Per day: ${total_pnl / max(n_days, 1):,.1f}')
    report.append('')

    report.append('DIRECTION:')
    for d in ['LONG', 'SHORT']:
        sub = df_trades[df_trades.direction == d]
        if len(sub) > 0:
            wr = (sub.pnl_ticks > 0).mean() * 100
            report.append(f'  {d}: {len(sub)} trades, WR={wr:.0f}%, '
                          f'avg={sub.pnl_ticks.mean():.1f}t, total=${sub.pnl_dollars.sum():,.1f}')

    report.append('')
    report.append('TOP TEMPLATES:')
    tid_grp = df_trades.groupby('template_id').agg(
        n=('pnl_ticks', 'count'),
        pnl=('pnl_dollars', 'sum'),
        wr=('pnl_ticks', lambda x: (x > 0).mean() * 100),
        avg=('pnl_ticks', 'mean'),
    ).sort_values('pnl', ascending=False)
    for tid, row in tid_grp.head(10).iterrows():
        report.append(f'  TID={tid}: n={row.n:.0f} WR={row.wr:.0f}% '
                      f'avg={row.avg:.1f}t total=${row.pnl:,.1f}')

    report.append('')
    report.append('COMPARISON:')
    report.append(f'  Blind 11-bar flip (OOS):     $7,967')
    report.append(f'  Current system (honest):     $1,844')
    report.append(f'  Template instructions (this): ${total_pnl:,.1f}')

    report_text = '\n'.join(report)
    print(report_text)

    # Save
    os.makedirs('reports/findings', exist_ok=True)
    with open('reports/findings/template_instruction_oos.txt', 'w') as f:
        f.write(report_text)
    df_trades.to_csv('reports/findings/template_instruction_oos.csv', index=False)
    print(f'\nSaved: reports/findings/template_instruction_oos.txt')
    print(f'Saved: reports/findings/template_instruction_oos.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS_OOS')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--match-threshold', type=float, default=3.0)
    args = parser.parse_args()
    run_test(args.data, args.checkpoint_dir, args.match_threshold)

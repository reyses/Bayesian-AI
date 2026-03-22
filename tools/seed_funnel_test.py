"""Seed Funnel Test — Progressive R-adjusted template matching on OOS.

At each bar, compares the rolling feature trajectory against IS-learned
template approach signatures. As more bars match, candidates narrow.
When the funnel collapses to one template, enter its direction + hold.

Usage:
    python tools/seed_funnel_test.py
    python tools/seed_funnel_test.py --data DATA/ATLAS_OOS --r2-threshold 0.6
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine

TICK_SIZE = 0.25
TICK_VALUE = 0.50
WINDOW = 10  # bars of lookback for trajectory matching

# Features to track in trajectory (from state dict keys in replay)
TRAJ_FEATURES = ['f_mom', 'z', 'vel', 'dmi_p', 'dmi_m', 'adx', 'coherence']


def build_template_trajectories(replay_path: str):
    """Build approach trajectories from IS trade replays.

    Each trade has 10 bars before entry. Extract the feature trajectory
    for those 10 bars, normalize (z-score per feature), and group by
    trade outcome (direction, PnL bucket) to create template signatures.
    """
    with open(replay_path) as f:
        replays = json.load(f)

    print(f'Building trajectories from {len(replays)} replays...')

    trajectories = []
    for t in replays:
        entry_bar = t['entry_bar']
        states = t['states']
        if entry_bar < WINDOW or len(states) < entry_bar:
            continue

        # Extract WINDOW bars before entry
        traj = np.zeros((WINDOW, len(TRAJ_FEATURES)))
        for j in range(WINDOW):
            s = states[entry_bar - WINDOW + j]
            for k, feat in enumerate(TRAJ_FEATURES):
                traj[j, k] = s.get(feat, 0.0)

        # Normalize: z-score per feature across the window
        means = traj.mean(axis=0)
        stds = traj.std(axis=0)
        stds[stds < 1e-8] = 1.0
        traj_normed = (traj - means) / stds

        trajectories.append({
            'traj': traj_normed,
            'traj_raw': traj,
            'means': means,
            'stds': stds,
            'side': t['side'],
            'pnl': t['actual_pnl'],
            'hold_bars': t['hold_bars'],
            'mfe': t.get('trade_mfe_ticks', 0),
        })

    print(f'Valid trajectories: {len(trajectories)}')

    # Cluster trajectories by outcome: win-long, win-short, lose-long, lose-short
    clusters = defaultdict(list)
    for t in trajectories:
        win = t['pnl'] > 0
        key = f"{'WIN' if win else 'LOSE'}_{t['side'].upper()}"
        clusters[key].append(t)

    # Build template signatures: average trajectory per cluster
    templates = {}
    for key, members in clusters.items():
        if len(members) < 5:
            continue
        avg_traj = np.mean([m['traj'] for m in members], axis=0)
        avg_hold = np.mean([m['hold_bars'] for m in members])
        avg_pnl = np.mean([m['pnl'] for m in members])
        direction = 'LONG' if 'LONG' in key else 'SHORT'
        is_winner = 'WIN' in key

        templates[key] = {
            'trajectory': avg_traj,
            'direction': direction,
            'is_winner': is_winner,
            'hold_bars_15s': int(avg_hold),
            'avg_pnl': avg_pnl,
            'n_members': len(members),
        }
        print(f'  {key}: {len(members)} members, hold={avg_hold:.0f} bars, '
              f'pnl=${avg_pnl:.1f}')

    return templates


def compute_r2(observed, template, k):
    """R-squared between last k bars of observed and first k bars of template.
    Computed per feature, returns mean R2 across features.
    """
    if k < 2:
        return 0.0

    obs = observed[-k:]  # (k, n_features)
    tmpl = template[:k]  # (k, n_features)

    r2_per_feat = []
    for f in range(obs.shape[1]):
        o = obs[:, f]
        t = tmpl[:, f]
        ss_res = np.sum((o - t) ** 2)
        ss_tot = np.sum((o - o.mean()) ** 2)
        if ss_tot < 1e-10:
            r2_per_feat.append(0.0)
        else:
            r2_per_feat.append(1.0 - ss_res / ss_tot)

    return np.mean(r2_per_feat)


def run_funnel(data_path: str, replay_path: str, r2_match: float = 0.5,
               r2_reject: float = 0.0):
    """Run progressive funnel matching on OOS data."""

    # Build template trajectories from IS
    templates = build_template_trajectories(replay_path)
    if not templates:
        print('No templates built')
        return

    # Only use WINNER templates for entry
    win_templates = {k: v for k, v in templates.items() if v['is_winner']}
    lose_templates = {k: v for k, v in templates.items() if not v['is_winner']}
    print(f'\nWinner templates: {len(win_templates)}')
    print(f'Loser templates: {len(lose_templates)} (used for rejection)')

    # Load OOS 15s data
    tf_dir = os.path.join(data_path, '15s')
    files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f'OOS bars: {len(df)}')

    # Compute states
    print('Computing states...')
    engine = StatisticalFieldEngine()
    states = engine.batch_compute_states(df)
    print(f'States: {len(states)}')

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Build feature matrix from states
    print('Extracting features...')
    feat_matrix = np.zeros((len(states), len(TRAJ_FEATURES)))
    for i, state in enumerate(states):
        if state is None:
            continue
        for k, feat in enumerate(TRAJ_FEATURES):
            feat_matrix[i, k] = getattr(state, feat if feat != 'z' else 'z_score',
                                         getattr(state, feat, 0.0))
    # Fix z_score mapping
    for i, state in enumerate(states):
        if state is not None:
            feat_matrix[i, 1] = getattr(state, 'z_score', 0.0)

    # Simulate funnel
    trades = []
    in_trade = False
    trade_entry_price = 0.0
    trade_dir = ''
    trade_exit_bar = 0
    trade_entry_bar = 0
    trade_template = ''
    trade_r2 = 0.0

    funnel_stats = {'matches': 0, 'no_match': 0,
                    'avg_candidates_10': [], 'avg_candidates_5': [], 'avg_candidates_1': []}

    print(f'Running funnel (r2_match={r2_match}, r2_reject={r2_reject})...')
    for i in tqdm(range(WINDOW, len(states), 4), desc='Funnel', miniters=500):
        # Exit check
        if in_trade and i >= trade_exit_bar:
            exit_price = closes[i]
            if trade_dir == 'LONG':
                pnl = (exit_price - trade_entry_price) / TICK_SIZE
            else:
                pnl = (trade_entry_price - exit_price) / TICK_SIZE
            trades.append({
                'entry_bar': trade_entry_bar, 'exit_bar': i,
                'direction': trade_dir, 'pnl_ticks': pnl,
                'pnl_dollars': pnl * TICK_VALUE,
                'hold_bars': i - trade_entry_bar,
                'template': trade_template, 'r2': trade_r2,
            })
            in_trade = False

        if in_trade:
            continue

        # Get current window, normalize
        window = feat_matrix[i - WINDOW:i].copy()
        w_means = window.mean(axis=0)
        w_stds = window.std(axis=0)
        w_stds[w_stds < 1e-8] = 1.0
        window_normed = (window - w_means) / w_stds

        # Score all winner templates at multiple K values
        best_key = None
        best_r2 = -1.0
        second_r2 = -1.0
        n_candidates = {10: 0, 5: 0, 1: 0}

        for key, tmpl in win_templates.items():
            # Progressive: check at K=3, K=6, K=10
            r2_full = compute_r2(window_normed, tmpl['trajectory'], WINDOW)
            r2_half = compute_r2(window_normed, tmpl['trajectory'], WINDOW // 2)

            # Only count as candidate if partial match holds
            if r2_half > r2_reject:
                n_candidates[5] += 1
            if r2_full > r2_reject:
                n_candidates[10] += 1

            if r2_full > best_r2:
                second_r2 = best_r2
                best_r2 = r2_full
                best_key = key
            elif r2_full > second_r2:
                second_r2 = r2_full

        # Check loser templates — if a loser matches better, skip
        loser_best_r2 = -1.0
        for key, tmpl in lose_templates.items():
            r2_full = compute_r2(window_normed, tmpl['trajectory'], WINDOW)
            if r2_full > loser_best_r2:
                loser_best_r2 = r2_full

        funnel_stats['avg_candidates_10'].append(n_candidates[10])
        funnel_stats['avg_candidates_5'].append(n_candidates[5])

        # Funnel collapsed?
        if (best_r2 >= r2_match
                and best_r2 > second_r2 * 1.3  # 30% better than runner-up
                and best_r2 > loser_best_r2):   # better than any loser
            tmpl = win_templates[best_key]
            trade_dir = tmpl['direction']
            trade_entry_price = closes[i]
            trade_entry_bar = i
            trade_exit_bar = i + tmpl['hold_bars_15s']
            trade_template = best_key
            trade_r2 = best_r2
            in_trade = True
            funnel_stats['matches'] += 1
        else:
            funnel_stats['no_match'] += 1

    # Results
    df_t = pd.DataFrame(trades)
    if len(df_t) == 0:
        print('No trades generated')
        return

    wins = df_t[df_t.pnl_ticks > 0]
    total_pnl = df_t.pnl_dollars.sum()
    n_days = len(df) / 2400

    report = []
    report.append('=' * 70)
    report.append('SEED FUNNEL TEST -- Progressive R2 Matching on OOS')
    report.append(f'R2 match threshold: {r2_match} | R2 reject: {r2_reject}')
    report.append('=' * 70)
    report.append('')
    report.append(f'Trades: {len(df_t)}')
    report.append(f'WR: {len(wins)/len(df_t)*100:.1f}%')
    report.append(f'Total PnL: {df_t.pnl_ticks.sum():.0f}t (${total_pnl:,.1f})')
    report.append(f'Avg PnL: {df_t.pnl_ticks.mean():.1f}t (${df_t.pnl_dollars.mean():.2f})')
    report.append(f'Avg R2 at entry: {df_t.r2.mean():.3f}')
    report.append(f'Avg hold: {df_t.hold_bars.mean():.0f} bars ({df_t.hold_bars.mean()*15/60:.1f} min)')
    report.append(f'Per day: ${total_pnl/max(n_days,1):,.1f}')
    report.append('')

    report.append('DIRECTION:')
    for d in ['LONG', 'SHORT']:
        sub = df_t[df_t.direction == d]
        if len(sub) > 0:
            wr = (sub.pnl_ticks > 0).mean() * 100
            report.append(f'  {d}: {len(sub)} trades, WR={wr:.0f}%, '
                          f'avg={sub.pnl_ticks.mean():.1f}t, total=${sub.pnl_dollars.sum():,.1f}')

    report.append('')
    report.append('FUNNEL STATS:')
    report.append(f'  Matches: {funnel_stats["matches"]}')
    report.append(f'  No match bars: {funnel_stats["no_match"]}')
    if funnel_stats['avg_candidates_10']:
        report.append(f'  Avg candidates at K=10: {np.mean(funnel_stats["avg_candidates_10"]):.1f}')
        report.append(f'  Avg candidates at K=5: {np.mean(funnel_stats["avg_candidates_5"]):.1f}')

    report.append('')
    report.append('BY TEMPLATE:')
    for key, grp in df_t.groupby('template'):
        wr = (grp.pnl_ticks > 0).mean() * 100
        report.append(f'  {key}: n={len(grp)} WR={wr:.0f}% avg={grp.pnl_ticks.mean():.1f}t '
                      f'r2={grp.r2.mean():.3f}')

    report.append('')
    report.append('COMPARISON:')
    report.append(f'  Blind 11-bar flip (OOS):      $7,967')
    report.append(f'  Current system (honest):      $1,844')
    report.append(f'  Seed funnel (this):           ${total_pnl:,.1f}')

    report_text = '\n'.join(report)
    print(report_text)

    os.makedirs('reports/findings', exist_ok=True)
    with open('reports/findings/seed_funnel_oos.txt', 'w') as f:
        f.write(report_text)
    df_t.to_csv('reports/findings/seed_funnel_oos.csv', index=False)
    print(f'\nSaved: reports/findings/seed_funnel_oos.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS_OOS')
    parser.add_argument('--replays', default='reports/trade_replays/is_replays.json')
    parser.add_argument('--r2-threshold', type=float, default=0.5,
                        help='R2 threshold for funnel collapse (match)')
    parser.add_argument('--r2-reject', type=float, default=0.0,
                        help='R2 below this = reject candidate')
    args = parser.parse_args()
    run_funnel(args.data, args.replays, args.r2_threshold, args.r2_reject)

"""Visualize Bayes exit oracle ON real tier trades for a single day.

For each trade in the iso pickle that opened on the chosen day:
    1. Plot entry/exit markers (actual)
    2. Simulate the Bayes exit oracle bar-by-bar
    3. Plot where the oracle WOULD have exited
    4. Show $ delta: oracle-exit P&L vs actual-exit P&L

USAGE
    python tools/chart_bayes_exit_oracle.py --day 2025_10_29 --tier FADE_CALM
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, time, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s, _load_tf_ohlcv
from training_iso_v2.filters.bayes_exit_oracle import BayesExitOracle


@dataclass
class FakeState:
    timestamp: float
    price: float
    v2: dict


@dataclass
class FakeTrade:
    entry_ts: float
    direction: str
    entry_price: float
    peak_pnl: float = 0.0


def load_tier_trades(tier: str, day: str):
    rows = []
    for split in ('IS', 'OOS'):
        path = f'training_iso_v2/output/{split.lower()}_{tier}.pkl'
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            trades = pickle.load(f)
        for t in trades:
            if t.entry_day == day:
                rows.append({
                    'direction': t.direction, 'entry_ts': int(t.entry_ts),
                    'exit_ts': int(t.exit_ts),
                    'entry_price': t.entry_price, 'exit_price': t.exit_price,
                    'pnl': t.pnl, 'peak_pnl': t.peak_pnl,
                    'exit_reason': t.exit_reason,
                })
    return rows


def load_v2_features_day(day: str):
    """Load L2_15m_vol_sigma_12 and other features at 5s grid."""
    layer_dir = 'DATA/ATLAS/FEATURES_5s_v2'
    df_5s = _load_5s(day)
    if df_5s.empty: return None
    close_df = df_5s[['timestamp', 'close']].copy()
    close_df['timestamp'] = close_df['timestamp'].astype(np.int64)
    for layer in ('L2_15m', 'L3_15m', 'L2_1h', 'L2_5m'):
        path = f'{layer_dir}/{layer}/{day}.parquet'
        if os.path.exists(path):
            f_df = pd.read_parquet(path)
            f_df['timestamp'] = f_df['timestamp'].astype(np.int64)
            close_df = pd.merge_asof(close_df.sort_values('timestamp'),
                                       f_df.sort_values('timestamp'),
                                       on='timestamp', direction='backward')
    return close_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_10_29')
    ap.add_argument('--tier', default='FADE_CALM')
    ap.add_argument('--out-dir', default='chart/bayes_framework')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades = load_tier_trades(args.tier, args.day)
    print(f'Loaded {len(trades)} {args.tier} trades on {args.day}')
    if not trades:
        print('No trades — pick a different day or tier'); return

    df = load_v2_features_day(args.day)
    if df is None or df.empty:
        print('No 5s/V2 data'); return
    ts_arr = df['timestamp'].values.astype(np.int64)
    close_arr = df['close'].values.astype(np.float64)
    v2_cols = [c for c in df.columns if c not in ('timestamp', 'close')]

    oracle = BayesExitOracle()

    # Simulate Bayes exit per trade
    sim_results = []
    for t in trades:
        # Walk bar-by-bar from entry to (latest possible exit, e.g., entry+3hr)
        i_entry = int(np.searchsorted(ts_arr, t['entry_ts']))
        i_max   = min(i_entry + (3 * 3600 // 5), len(ts_arr) - 1)
        fake_trade = FakeTrade(entry_ts=t['entry_ts'],
                                 direction=t['direction'],
                                 entry_price=t['entry_price'],
                                 peak_pnl=0.0)
        bayes_exit_ts = None
        bayes_exit_reason = None
        peak_pnl_running = 0.0
        for i in range(i_entry + 1, i_max):
            cur_price = close_arr[i]
            cur_pnl = ((cur_price - t['entry_price']) if t['direction']=='long'
                        else (t['entry_price'] - cur_price))
            peak_pnl_running = max(peak_pnl_running, cur_pnl)
            fake_trade.peak_pnl = peak_pnl_running
            v2_dict = {c: df[c].iloc[i] for c in v2_cols
                        if not pd.isna(df[c].iloc[i])}
            state = FakeState(timestamp=ts_arr[i], price=cur_price, v2=v2_dict)
            should_exit, reason = oracle.should_exit(state, fake_trade)
            if should_exit:
                bayes_exit_ts = int(ts_arr[i])
                bayes_exit_reason = reason
                bayes_exit_price = cur_price
                break
        if bayes_exit_ts is None:
            bayes_exit_ts = int(ts_arr[i_max])
            bayes_exit_reason = 'time_cap'
            bayes_exit_price = close_arr[i_max]
        bayes_pnl = ((bayes_exit_price - t['entry_price']) if t['direction']=='long'
                       else (t['entry_price'] - bayes_exit_price))
        # MNQ pt value
        bayes_pnl_dollar = bayes_pnl * 2.0
        sim_results.append({
            **t,
            'bayes_exit_ts': bayes_exit_ts,
            'bayes_exit_price': bayes_exit_price,
            'bayes_exit_reason': bayes_exit_reason,
            'bayes_pnl': round(bayes_pnl_dollar, 2),
            'delta_vs_actual': round(bayes_pnl_dollar - t['pnl'], 2),
        })

    sim_df = pd.DataFrame(sim_results)
    sim_df.to_csv(os.path.join(args.out_dir,
                                  f'bayes_exit_sim_{args.tier}_{args.day}.csv'),
                    index=False)
    print(f'\nResults:')
    print(sim_df[['entry_ts','direction','entry_price','pnl','exit_reason',
                    'bayes_pnl','bayes_exit_reason','delta_vs_actual']]
            .to_string(index=False))
    actual_sum = sim_df['pnl'].sum()
    bayes_sum  = sim_df['bayes_pnl'].sum()
    print(f'\nTotal actual P&L: ${actual_sum:+.0f}')
    print(f'Total bayes P&L:  ${bayes_sum:+.0f}')
    print(f'Delta:            ${bayes_sum - actual_sum:+.0f}')

    # ===== CHART =====
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_arr]
    fig, ax = plt.subplots(figsize=(22, 9))
    ax.plot(dt, close_arr, color='black', lw=0.5, alpha=0.85, label='5s close')

    for _, t in sim_df.iterrows():
        entry_dt = datetime.fromtimestamp(int(t['entry_ts']), tz=timezone.utc)
        actual_exit_dt = datetime.fromtimestamp(int(t['exit_ts']), tz=timezone.utc)
        bayes_exit_dt = datetime.fromtimestamp(int(t['bayes_exit_ts']),
                                                  tz=timezone.utc)
        # Entry
        col = '#43A047' if t['direction']=='long' else '#E53935'
        marker = '^' if t['direction']=='long' else 'v'
        ax.scatter([entry_dt], [t['entry_price']], color=col, s=90, marker=marker,
                     edgecolor='black', zorder=5, label=f'entry ({t["direction"]})'
                     if _ == sim_df.index[0] else None)
        # Actual exit
        ax.scatter([actual_exit_dt], [t['exit_price']], color=col, s=70,
                     marker='X', edgecolor='black', zorder=5,
                     label='actual exit' if _ == sim_df.index[0] else None)
        # Bayes exit
        ax.scatter([bayes_exit_dt], [t['bayes_exit_price']], color='#FFB300',
                     s=110, marker='o', edgecolor='black', zorder=6,
                     label='bayes exit' if _ == sim_df.index[0] else None)
        # Line from entry to bayes exit
        ax.plot([entry_dt, bayes_exit_dt],
                  [t['entry_price'], t['bayes_exit_price']],
                  color='#FFB300', lw=1.5, alpha=0.6, linestyle=':')
        # Annotation with delta
        delta = t['delta_vs_actual']
        delta_col = '#43A047' if delta > 0 else '#E53935'
        ax.text(bayes_exit_dt, t['bayes_exit_price'],
                  f'  Δ${delta:+.0f}', fontsize=7, color=delta_col,
                  va='center')

    ax.set_title(f'{args.day} — {args.tier} trades + BAYES EXIT ORACLE overlay\n'
                   f'Actual total: ${actual_sum:+.0f}   '
                   f'Bayes total: ${bayes_sum:+.0f}   '
                   f'Delta: ${bayes_sum-actual_sum:+.0f}',
                   fontsize=11)
    ax.set_xlabel('time (UTC)'); ax.set_ylabel('price')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.tight_layout()
    out = os.path.join(args.out_dir,
                         f'bayes_exit_overlay_{args.tier}_{args.day}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out}')


if __name__ == '__main__':
    main()

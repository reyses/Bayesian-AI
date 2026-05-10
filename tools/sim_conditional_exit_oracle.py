"""Simulate the conditional Bayes exit oracle on a tier's actual trades.

For each trade in iso pickle:
    1. Walk every 5s bar from entry to exit
    2. Update peak tracker
    3. Query conditional exit oracle (bayes_conditional_exit module)
    4. Record the first bar where oracle says EXIT
    5. Compute bayes-exit PnL vs actual PnL

Compare aggregate $ delta and per-trade outcomes.

USAGE
    python tools/sim_conditional_exit_oracle.py --day 2025_10_29 --tier FADE_CALM
    python tools/sim_conditional_exit_oracle.py --tier FADE_CALM --all-days
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s
from training_iso_v2.filters.bayes_conditional_exit import BayesConditionalExit


def simulate_trade(trade, ts_arr, close_arr, oracle):
    """Walk this trade bar-by-bar, return (bayes_exit_ts, bayes_exit_price,
    bayes_exit_reason, bayes_pnl)."""
    i_entry = int(np.searchsorted(ts_arr, trade.entry_ts))
    i_exit  = int(np.searchsorted(ts_arr, trade.exit_ts))
    if i_exit <= i_entry: return None
    oracle.reset(entry_ts=int(trade.entry_ts))
    for k in range(i_exit - i_entry + 1):
        i = i_entry + k
        cur_price = close_arr[i]
        cur_pnl = ((cur_price - trade.entry_price) if trade.direction=='long'
                    else (trade.entry_price - cur_price))
        cur_pnl_dollar = cur_pnl * 2.0  # MNQ pt value
        should_exit, reason, p_final = oracle.update_and_query(
            current_pnl=cur_pnl_dollar, current_ts=int(ts_arr[i]))
        if should_exit:
            return {
                'bayes_exit_ts': int(ts_arr[i]),
                'bayes_exit_price': float(cur_price),
                'bayes_exit_reason': reason,
                'bayes_pnl': round(cur_pnl_dollar, 2),
                'p_final_at_exit': round(p_final, 3),
            }
    # Never fired — use actual exit
    return {
        'bayes_exit_ts': int(trade.exit_ts),
        'bayes_exit_price': float(trade.exit_price),
        'bayes_exit_reason': 'oracle_never_fired',
        'bayes_pnl': round(float(trade.pnl), 2),
        'p_final_at_exit': np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None)
    ap.add_argument('--tier', default='FADE_CALM')
    ap.add_argument('--all-days', action='store_true')
    ap.add_argument('--out-dir', default='reports/findings/segments/sim_conditional_exit')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    oracle = BayesConditionalExit(tier=args.tier)
    print(f'Tier: {args.tier}  threshold: {oracle.threshold}')

    # Load all trades for this tier
    all_trades = []
    for split in ('IS', 'OOS'):
        path = f'training_iso_v2/output/{split.lower()}_{args.tier}.pkl'
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            trades = pickle.load(f)
        for t in trades:
            if args.day and t.entry_day != args.day: continue
            all_trades.append((split, t))
    print(f'Trades to simulate: {len(all_trades):,}')

    # Group by day
    by_day = {}
    for split, t in all_trades:
        by_day.setdefault(t.entry_day, []).append((split, t))

    rows = []
    for day in tqdm(sorted(by_day.keys()), desc='days'):
        df_5s = _load_5s(day)
        if df_5s.empty: continue
        ts_arr = df_5s['timestamp'].values.astype(np.int64)
        close_arr = df_5s['close'].values.astype(np.float64)
        for split, t in by_day[day]:
            sim = simulate_trade(t, ts_arr, close_arr, oracle)
            if sim is None: continue
            rows.append({
                'day': day, 'split': split,
                'direction': t.direction,
                'entry_ts': int(t.entry_ts),
                'actual_exit_ts': int(t.exit_ts),
                'entry_price': t.entry_price,
                'actual_exit_price': t.exit_price,
                'actual_pnl': float(t.pnl),
                'actual_exit_reason': t.exit_reason,
                **sim,
                'delta_vs_actual': round(sim['bayes_pnl'] - float(t.pnl), 2),
            })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir,
                              f'sim_{args.tier}_{args.day or "ALL"}.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nResults -> {out_csv}')

    # Aggregate
    actual_sum = df['actual_pnl'].sum()
    bayes_sum = df['bayes_pnl'].sum()
    print(f'\nActual total P&L: ${actual_sum:+.0f}')
    print(f'Bayes total P&L:  ${bayes_sum:+.0f}')
    print(f'Delta (Bayes - Actual): ${bayes_sum - actual_sum:+.0f}')
    n_fired = int((df['bayes_exit_reason'] != 'oracle_never_fired').sum())
    print(f'Bayes oracle fired on {n_fired}/{len(df)} trades ({100*n_fired/len(df):.1f}%)')

    if n_fired > 0:
        fired = df[df['bayes_exit_reason'] != 'oracle_never_fired']
        print(f'\nOf the {len(fired)} trades oracle fired on:')
        wins = (fired['delta_vs_actual'] > 0).sum()
        losses = (fired['delta_vs_actual'] < 0).sum()
        ties = (fired['delta_vs_actual'] == 0).sum()
        print(f'  Better than actual: {wins}  ({100*wins/len(fired):.1f}%)')
        print(f'  Worse than actual:  {losses} ({100*losses/len(fired):.1f}%)')
        print(f'  Tie:                {ties}')
        print(f'  Mean delta: ${fired["delta_vs_actual"].mean():+.2f}')
        print(f'  Median delta: ${fired["delta_vs_actual"].median():+.2f}')

    # Per-split (IS vs OOS) breakdown
    print(f'\n--- Per-split breakdown ---')
    for split, g in df.groupby('split'):
        actual_s = g['actual_pnl'].sum()
        bayes_s = g['bayes_pnl'].sum()
        print(f'  {split}: n={len(g):>5d}  actual=${actual_s:+.0f}  '
               f'bayes=${bayes_s:+.0f}  delta=${bayes_s-actual_s:+.0f}')

    # Chart (single-day only)
    if args.day and not df.empty:
        df_5s = _load_5s(args.day)
        ts = df_5s['timestamp'].values.astype(np.int64)
        dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
        close = df_5s['close'].values
        fig, ax = plt.subplots(figsize=(22, 9))
        ax.plot(dt, close, color='black', lw=0.5, alpha=0.85, label='5s close')
        for i, t in df.iterrows():
            entry_dt = datetime.fromtimestamp(int(t['entry_ts']), tz=timezone.utc)
            actual_dt = datetime.fromtimestamp(int(t['actual_exit_ts']), tz=timezone.utc)
            bayes_dt = datetime.fromtimestamp(int(t['bayes_exit_ts']), tz=timezone.utc)
            col = '#43A047' if t['direction']=='long' else '#E53935'
            marker = '^' if t['direction']=='long' else 'v'
            ax.scatter([entry_dt], [t['entry_price']], color=col, s=80, marker=marker,
                         edgecolor='black', zorder=5)
            ax.scatter([actual_dt], [t['actual_exit_price']],
                         color='#3F51B5', s=55, marker='X',
                         edgecolor='black', zorder=5)
            ax.scatter([bayes_dt], [t['bayes_exit_price']],
                         color='#FFB300', s=70, marker='o',
                         edgecolor='black', zorder=6)
            delta = t['delta_vs_actual']
            delta_col = '#43A047' if delta > 0 else '#E53935' if delta < 0 else '#999'
            ax.text(bayes_dt, t['bayes_exit_price'], f' Δ${delta:+.0f}',
                      fontsize=6, color=delta_col, va='center')
        # Legend
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([],[], marker='^', color='w', mfc='#43A047', mec='black',
                     markersize=10, label='long entry'),
            Line2D([],[], marker='v', color='w', mfc='#E53935', mec='black',
                     markersize=10, label='short entry'),
            Line2D([],[], marker='X', color='w', mfc='#3F51B5', mec='black',
                     markersize=10, label='actual exit'),
            Line2D([],[], marker='o', color='w', mfc='#FFB300', mec='black',
                     markersize=10, label='Bayes conditional exit'),
        ], loc='best', fontsize=9)
        ax.set_title(f'{args.day} — {args.tier} trades + BAYES CONDITIONAL EXIT\n'
                       f'n={len(df)}  actual=${actual_sum:+.0f}  '
                       f'bayes=${bayes_sum:+.0f}  delta=${bayes_sum-actual_sum:+.0f}',
                       fontsize=11)
        ax.set_xlabel('time (UTC)'); ax.set_ylabel('price')
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.tight_layout()
        out_png = os.path.join(args.out_dir,
                                  f'sim_overlay_{args.tier}_{args.day}.png')
        plt.savefig(out_png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()

"""Per-tier exit-quality analysis — does the prob table help exit better?

For each completed trade:
    1. Walk every 5s bar between entry and exit
    2. Track peak P&L reached AT ANY POINT during the hold
    3. Compute giveback = peak_pnl - actual_exit_pnl
    4. Compute peak_horizon = (peak_ts - entry_ts) in minutes
    5. Identify what direction price went after the peak (reverse vs continue)

Then aggregate:
    - Mean giveback per tier — large → room for improvement
    - Peak-horizon distribution per tier — informs cell-specific exit timing
    - For each trade, would a Bayes-table exit at peak_horizon have done better?

The result tells us:
    - If tier's actual exit IS near the peak → no exit-oracle help possible
    - If actual exit gives back substantial $ → exit oracle could capture more

USAGE
    python tools/tier_exit_giveback_analysis.py
    python tools/tier_exit_giveback_analysis.py --tier FADE_CALM
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s


TIERS = ['FADE_CALM','FADE_MOMENTUM','FADE_AGAINST','RIDE_CALM','RIDE_MOMENTUM',
         'RIDE_AGAINST','KILL_SHOT','CASCADE','FREIGHT_TRAIN','FADE_AT_BAND',
         'NMP_FADE_RAW','NMP_RIDE_RAW']


def analyze_trade(trade, ts_arr, close_arr) -> dict:
    """Walk this trade bar-by-bar; return peak P&L stats."""
    i_entry = int(np.searchsorted(ts_arr, trade.entry_ts))
    i_exit  = int(np.searchsorted(ts_arr, trade.exit_ts))
    if i_exit <= i_entry: return None
    window = close_arr[i_entry:i_exit+1]
    if trade.direction == 'long':
        pnl_path = window - trade.entry_price
    else:
        pnl_path = trade.entry_price - window
    # Convert pts to $ (MNQ: $2/pt)
    pnl_path_dollar = pnl_path * 2.0
    peak_idx = int(np.argmax(pnl_path_dollar))
    peak_pnl = float(pnl_path_dollar[peak_idx])
    peak_ts = int(ts_arr[i_entry + peak_idx])
    actual_pnl_dollar = float(trade.pnl)
    # giveback = how much $ was given back from peak to actual exit
    giveback = peak_pnl - actual_pnl_dollar
    peak_horizon_s = peak_ts - int(trade.entry_ts)
    return {
        'peak_pnl': round(peak_pnl, 2),
        'actual_pnl': round(actual_pnl_dollar, 2),
        'giveback': round(giveback, 2),
        'peak_horizon_min': round(peak_horizon_s / 60.0, 2),
        'hold_min': round((trade.exit_ts - trade.entry_ts) / 60.0, 2),
        'capture_pct': round(100 * actual_pnl_dollar / peak_pnl, 1) if peak_pnl > 0
                        else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default=None,
                     help='Single tier; default = ALL')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/tier_exit_giveback')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tiers = [args.tier] if args.tier else TIERS
    all_rows = []

    print(f'Loading trades and walking each bar-by-bar...')
    for tier in tiers:
        for split in ('IS', 'OOS'):
            path = f'training_iso_v2/output/{split.lower()}_{tier}.pkl'
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                trades = pickle.load(f)
            # Group by day to load 5s only once per day
            by_day = {}
            for t in trades:
                by_day.setdefault(t.entry_day, []).append(t)
            for day, day_trades in tqdm(by_day.items(),
                                          desc=f'  {tier}/{split}'):
                df_5s = _load_5s(day)
                if df_5s.empty: continue
                ts_arr = df_5s['timestamp'].values.astype(np.int64)
                close_arr = df_5s['close'].values.astype(np.float64)
                for t in day_trades:
                    out = analyze_trade(t, ts_arr, close_arr)
                    if out is None: continue
                    all_rows.append({
                        'tier': tier, 'split': split, 'day': day,
                        'direction': t.direction,
                        'entry_price': t.entry_price,
                        'exit_reason': t.exit_reason,
                        **out,
                    })

    df = pd.DataFrame(all_rows)
    if df.empty: print('no data'); return
    df.to_csv(os.path.join(args.out_dir, 'per_trade_giveback.csv'), index=False)
    print(f'\nAnalyzed {len(df):,} trades')

    # Aggregate per tier
    print(f'\n=== PER-TIER EXIT-QUALITY SUMMARY ===')
    rows = []
    for tier, g in df.groupby('tier'):
        n = len(g)
        pos_peaks = g[g['peak_pnl'] > 0]
        n_pos = len(pos_peaks)
        if n_pos == 0:
            rows.append({'tier': tier, 'n': n,
                         'pct_with_pos_peak': 0,
                         'mean_giveback': 0, 'median_giveback': 0,
                         'mean_capture_pct': 0,
                         'mean_peak_horizon_min': 0,
                         'large_giveback_count': 0,
                         'large_giveback_pct': 0})
            continue
        rows.append({
            'tier': tier, 'n': n,
            'pct_with_pos_peak':       round(100 * n_pos / n, 1),
            'mean_giveback':           round(float(pos_peaks['giveback'].mean()), 2),
            'median_giveback':         round(float(pos_peaks['giveback'].median()), 2),
            'mean_capture_pct':        round(float(pos_peaks['capture_pct'].mean()), 1),
            'mean_peak_horizon_min':   round(float(pos_peaks['peak_horizon_min'].mean()), 2),
            'median_peak_horizon_min': round(float(pos_peaks['peak_horizon_min'].median()), 2),
            'large_giveback_count':    int((pos_peaks['giveback'] >= 10).sum()),
            'large_giveback_pct':      round(100 * (pos_peaks['giveback'] >= 10).sum() / n_pos, 1),
        })
    summ = pd.DataFrame(rows).sort_values('mean_giveback', ascending=False)
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(args.out_dir, 'tier_summary.csv'), index=False)

    # Theoretical max P&L per tier (if we always exited at peak)
    print(f'\n=== THEORETICAL MAX P&L (perfect exit at peak) per tier ===')
    perfect_rows = []
    for tier, g in df.groupby('tier'):
        actual_sum  = float(g['actual_pnl'].sum())
        peak_sum    = float(g['peak_pnl'].sum())
        # 'realistic' bayes exit: capture, say, 70% of peak via trail
        bayes_70_sum = float((g['peak_pnl'] * 0.7).sum())
        perfect_rows.append({
            'tier': tier, 'n': len(g),
            'actual_total': round(actual_sum, 0),
            'peak_total':   round(peak_sum, 0),
            'bayes_70pct_peak': round(bayes_70_sum, 0),
            'uplift_if_70pct_peak_vs_actual': round(bayes_70_sum - actual_sum, 0),
        })
    perfect_df = pd.DataFrame(perfect_rows).sort_values(
        'uplift_if_70pct_peak_vs_actual', ascending=False)
    print(perfect_df.to_string(index=False))
    perfect_df.to_csv(os.path.join(args.out_dir, 'perfect_exit_uplift.csv'), index=False)

    # CHART
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    # Bar: mean giveback per tier
    ax = axes[0][0]
    s = summ.set_index('tier')['mean_giveback'].sort_values(ascending=True)
    ax.barh(s.index, s.values, color='#FB8C00', alpha=0.85)
    ax.set_xlabel('mean giveback $ per trade')
    ax.set_title('Mean giveback by tier  (higher = exits leave $ on table)')
    ax.grid(True, axis='x', alpha=0.3)

    # Bar: % large givebacks
    ax = axes[0][1]
    s2 = summ.set_index('tier')['large_giveback_pct'].sort_values(ascending=True)
    ax.barh(s2.index, s2.values, color='#E53935', alpha=0.85)
    ax.set_xlabel('% trades with giveback >= $10')
    ax.set_title('Frequency of large givebacks  (higher = more room for improvement)')
    ax.grid(True, axis='x', alpha=0.3)

    # Bar: capture % per tier
    ax = axes[1][0]
    s3 = summ.set_index('tier')['mean_capture_pct'].sort_values(ascending=False)
    colors = ['#43A047' if v >= 70 else '#FB8C00' if v >= 40 else '#E53935' for v in s3.values]
    ax.barh(s3.index, s3.values, color=colors, alpha=0.85)
    ax.set_xlabel('capture % (actual/peak)')
    ax.set_title('How much of the peak P&L tier exits actually capture\n'
                  'green=>=70%  orange=40-70%  red=<40%')
    ax.axvline(70, color='black', ls='--', lw=0.6)
    ax.grid(True, axis='x', alpha=0.3)

    # Bar: uplift if we captured 70% of peak
    ax = axes[1][1]
    s4 = perfect_df.set_index('tier')['uplift_if_70pct_peak_vs_actual'].sort_values(ascending=True)
    colors = ['#43A047' if v > 0 else '#E53935' for v in s4.values]
    ax.barh(s4.index, s4.values, color=colors, alpha=0.85)
    ax.set_xlabel('$ uplift if 70% of peak captured (vs actual)')
    ax.set_title('Total $ uplift over entire IS+OOS if exits captured 70% of peak\n'
                  'positive = bayes exit oracle has room to help')
    ax.axvline(0, color='black', lw=0.6)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'exit_giveback_analysis.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()

"""Per-tier CONDITIONAL exit probability table — fixed framing.

Reframing: the exit question is NOT 'what happens in next 60 min' but
'given my position is currently at +$X, peak so far is +$Y, time-in-trade
is t_min — what's the probability of reaching a NEW HIGHER peak before
the trade closes naturally?'

For each completed trade, we walk bar-by-bar and label each bar with:
    t_in_trade_min        time since entry
    current_pnl           P&L right now
    peak_pnl_so_far       max P&L reached in this trade so far
    time_since_peak_min   how long since peak was hit
    capture_ratio         current_pnl / peak_pnl_so_far
    is_peak_final         1 if no higher peak will ever occur in this trade
                          0 if a new higher peak will be reached later

Then aggregate per (tier, time_since_peak_bucket, capture_ratio_bucket) →
    P(is_peak_final | state) = P(NO new peak coming) = EXIT signal strength

Exit rule:  exit when P_is_peak_final >= threshold (e.g. 0.7)

Per-tier table calibrated to the TIER'S OWN trade timescale, not a
universal 60-min horizon.

USAGE
    python tools/tier_conditional_exit_table.py
    python tools/tier_conditional_exit_table.py --tier RIDE_AGAINST
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


def walk_trade(trade, ts_arr, close_arr) -> list[dict]:
    """For each bar of this trade, emit a labeled state row."""
    i_entry = int(np.searchsorted(ts_arr, trade.entry_ts))
    i_exit  = int(np.searchsorted(ts_arr, trade.exit_ts))
    if i_exit <= i_entry: return []
    bars = i_exit - i_entry + 1
    # Path in $
    window = close_arr[i_entry:i_exit+1]
    if trade.direction == 'long':
        pnl_path = (window - trade.entry_price) * 2.0
    else:
        pnl_path = (trade.entry_price - window) * 2.0
    rows = []
    running_peak = pnl_path[0]
    running_peak_idx = 0
    for k in range(bars):
        current_pnl = float(pnl_path[k])
        if current_pnl > running_peak:
            running_peak = current_pnl
            running_peak_idx = k
        # Did a higher peak occur AFTER this bar?
        if k < bars - 1:
            future_max = float(np.max(pnl_path[k+1:]))
            is_peak_final = int(future_max <= running_peak)
        else:
            is_peak_final = 1  # last bar
        rows.append({
            't_in_trade_s':     int((ts_arr[i_entry + k] - trade.entry_ts)),
            't_since_peak_s':   int((ts_arr[i_entry + k] - ts_arr[i_entry + running_peak_idx])),
            'current_pnl':      round(current_pnl, 2),
            'peak_pnl_so_far':  round(running_peak, 2),
            'capture_ratio':    round(current_pnl / running_peak, 3)
                                if running_peak > 0 else 0.0,
            'is_peak_final':    is_peak_final,
        })
    return rows


def bucket_time(s: float, bins: list = (5, 15, 30, 60, 120, 300, 900)) -> int:
    """Time-in-seconds → small int bucket."""
    for i, b in enumerate(bins):
        if s <= b: return i
    return len(bins)


def bucket_capture(r: float) -> int:
    """capture_ratio → bin: 0 = neg/zero, 1 = 0-30%, 2 = 30-60%, 3 = 60-90%,
    4 = 90-100% (at peak)."""
    if r <= 0: return 0
    if r < 0.3: return 1
    if r < 0.6: return 2
    if r < 0.9: return 3
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default=None)
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/tier_conditional_exit')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tiers = [args.tier] if args.tier else TIERS

    all_rows = []
    print('Walking each trade bar-by-bar...')
    for tier in tiers:
        for split in ('IS', 'OOS'):
            path = f'training_iso_v2/output/{split.lower()}_{tier}.pkl'
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                trades = pickle.load(f)
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
                    rows = walk_trade(t, ts_arr, close_arr)
                    for r in rows:
                        r.update({'tier': tier, 'split': split})
                        all_rows.append(r)

    df = pd.DataFrame(all_rows)
    if df.empty: print('no data'); return
    print(f'\nTotal bar-states recorded: {len(df):,} across {df["tier"].nunique()} tiers')

    # Build conditional exit table per tier
    print(f'\n=== CONDITIONAL EXIT TABLE: P(is_peak_final | tier, t_since_peak, capture_ratio) ===')
    df['t_since_peak_bucket'] = df['t_since_peak_s'].apply(bucket_time)
    df['capture_bucket']      = df['capture_ratio'].apply(bucket_capture)
    table_rows = []
    for (tier, t_bucket, cap_bucket), g in df.groupby(['tier','t_since_peak_bucket','capture_bucket']):
        is_g = g[g['split']=='IS']
        oos_g = g[g['split']=='OOS']
        n_is = len(is_g); n_oos = len(oos_g)
        if n_is < 50: continue
        p_final_is  = float(is_g['is_peak_final'].mean())
        p_final_oos = float(oos_g['is_peak_final'].mean()) if n_oos >= 20 else np.nan
        if pd.notna(p_final_oos):
            sign_match = int((p_final_is >= 0.5) == (p_final_oos >= 0.5))
        else:
            sign_match = -1
        table_rows.append({
            'tier': tier, 't_since_peak_bucket': int(t_bucket),
            'capture_bucket': int(cap_bucket),
            'n_is': n_is, 'n_oos': n_oos,
            'P_is_peak_final_IS':  round(p_final_is, 3),
            'P_is_peak_final_OOS': round(p_final_oos, 3) if pd.notna(p_final_oos) else np.nan,
            'sign_match': sign_match,
        })
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(os.path.join(args.out_dir, 'conditional_exit_table.csv'),
                     index=False)
    print(f'Cells with n_is >= 50: {len(table_df)}')

    # ─── Per-tier extract: cells where P_is_peak_final crosses important thresholds
    print(f'\n=== HIGH-CONFIDENCE EXIT CELLS (P_final >= 0.70 IS + OOS sign match) ===')
    high_exit = table_df[(table_df['P_is_peak_final_IS'] >= 0.70) &
                           (table_df['sign_match'] == 1) &
                           (table_df['n_oos'] >= 20)].copy()
    if not high_exit.empty:
        cols = ['tier','t_since_peak_bucket','capture_bucket','n_is','n_oos',
                 'P_is_peak_final_IS','P_is_peak_final_OOS']
        print(high_exit[cols].sort_values('P_is_peak_final_IS', ascending=False)
                  .head(30).to_string(index=False))
        high_exit.to_csv(os.path.join(args.out_dir, 'high_confidence_exit_cells.csv'),
                          index=False)

    # ─── Per-tier summary: at what capture-ratio do we hit P_final >= 0.7?
    print(f'\n=== PER-TIER EXIT RECOMMENDATION (at-peak cells) ===')
    summary_rows = []
    for tier in tiers:
        # at-peak bucket (capture_bucket=4)
        at_peak = table_df[(table_df['tier']==tier) & (table_df['capture_bucket']==4)]
        if at_peak.empty: continue
        # Sort by t_since_peak — exit recommendation rises with time
        at_peak = at_peak.sort_values('t_since_peak_bucket')
        # Find first bucket where P_final >= 0.5
        first_exit_bucket = None
        for _, r in at_peak.iterrows():
            if r['P_is_peak_final_IS'] >= 0.5:
                first_exit_bucket = int(r['t_since_peak_bucket']); break
        summary_rows.append({
            'tier': tier,
            'at_peak_buckets': len(at_peak),
            'first_p_final_ge_50pct_bucket': first_exit_bucket,
            'max_p_final': round(float(at_peak['P_is_peak_final_IS'].max()), 3),
            'mean_n_is': int(at_peak['n_is'].mean()),
        })
    summ = pd.DataFrame(summary_rows)
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(args.out_dir, 'per_tier_exit_recommendation.csv'), index=False)

    # Heatmap chart per tier (capture_bucket × t_since_peak_bucket)
    n_tiers = len(tiers)
    cols_grid = 3
    rows_grid = int(np.ceil(n_tiers / cols_grid))
    fig, axes = plt.subplots(rows_grid, cols_grid,
                              figsize=(cols_grid * 6, rows_grid * 4),
                              squeeze=False)
    axes_flat = axes.ravel()
    for ai, tier in enumerate(tiers):
        ax = axes_flat[ai]
        sub = table_df[table_df['tier']==tier]
        if sub.empty:
            ax.set_title(f'{tier} (no data)', fontsize=10); ax.set_visible(False); continue
        pivot = sub.pivot_table(index='capture_bucket',
                                  columns='t_since_peak_bucket',
                                  values='P_is_peak_final_IS')
        if pivot.empty: continue
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f'{tier}\nP(peak_final | t_since_peak, capture)', fontsize=9)
        ax.set_xlabel('t_since_peak bucket', fontsize=8)
        ax.set_ylabel('capture_bucket', fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    color = 'white' if v > 0.6 else 'black'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                              fontsize=6, color=color)
    for j in range(n_tiers, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle('CONDITIONAL EXIT TABLE per tier: P(current peak IS final peak | state)\n'
                  'Higher P = more confident to EXIT NOW  '
                  '(t_since_peak buckets: 0=≤5s, 1=≤15s, 2=≤30s, 3=≤60s, 4=≤120s, 5=≤300s, 6=≤900s, 7=>900s)\n'
                  '(capture buckets: 0=neg, 1=0-30%, 2=30-60%, 3=60-90%, 4=90-100% at peak)',
                  fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(args.out_dir, 'conditional_exit_heatmap.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nHeatmap -> {out_png}')


if __name__ == '__main__':
    main()

"""Catastrophic-event HARVEST analysis.

Reframes the cat-event risk from defensive (avoid) to offensive (harness).
If P(cat) is 40% at Tuesday UTC 1, the EV of positioning for the event is:
    EV = P(cat) * E[|move| | cat] * direction_skew  -  P(no cat) * baseline_drift

This tool computes per-cell:
    1. Direction skew: P(above_high) vs P(below_low) among cat events
    2. Mean signed magnitude (z and dollars)
    3. EV of a long position, short position, and 'straddle' (both stops)
    4. The most exploitable cells (high P_cat + directional bias)

USAGE
    python tools/bayes_table_cat_harvest.py
    python tools/bayes_table_cat_harvest.py --cat-threshold 6.0
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# MNQ: tick = 0.25, tick value = $0.50; so 1 point = $2
MNQ_PT_VALUE = 2.0
DOW_MAP = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}


def beta_p(k, n):
    a, b = k+1, n-k+1
    return (float(a/(a+b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro-csv',
                     default='reports/findings/band_touch_aggregation/macro_events_1h_hl.csv')
    ap.add_argument('--cat-threshold', type=float, default=6.0)
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_cat_harvest')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    evs = pd.read_csv(args.macro_csv)
    evs['dow_str'] = evs['dow']  # already string
    evs['date'] = pd.to_datetime(evs['day'].str.replace('_', '-'))
    cat = evs[evs['max_abs_z'] >= args.cat_threshold].copy()
    print(f'Cat events (|max_z|>={args.cat_threshold}): {len(cat)}')
    print(f'  Directional split: above={(cat["side"]=="above").sum()} ({100*(cat["side"]=="above").mean():.1f}%)')
    print(f'                     below={(cat["side"]=="below").sum()} ({100*(cat["side"]=="below").mean():.1f}%)')

    # SE_close approximated as max_abs_z / signed_max_z ratio mismatch -> use macro_events stats
    # We don't have SE_close per event directly; approximate excursion magnitude using max_abs_z
    # Actual $ move computed from event_substrate which has fwd returns
    # For directional bias, just use side.

    # ===== Directional split per DOW =====
    print(f'\n=== DIRECTIONAL SPLIT per DOW (|max_z|>={args.cat_threshold}) ===')
    rows_dir = []
    for dow, g in cat.groupby('dow_str'):
        n = len(g)
        n_above = int((g['side']=='above').sum())
        n_below = int((g['side']=='below').sum())
        # Beta posterior on P(above)
        p_above, lo_a, hi_a = beta_p(n_above, n)
        mean_z = float(g['signed_max_z'].mean())
        rows_dir.append({
            'dow': dow, 'n_cat_events': n,
            'n_above': n_above, 'n_below': n_below,
            'P_above': round(p_above, 4),
            'cilo': round(lo_a, 4), 'cihi': round(hi_a, 4),
            'mean_signed_z': round(mean_z, 2),
            'mean_abs_z': round(float(g['max_abs_z'].mean()), 2),
        })
    dir_df = pd.DataFrame(rows_dir)
    dir_df = dir_df.set_index('dow').reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']).dropna(how='all').reset_index()
    print(dir_df.to_string(index=False))
    dir_df.to_csv(os.path.join(args.out_dir, 'dir_skew_by_dow.csv'), index=False)

    # ===== Directional split per (DOW, TOD) =====
    print(f'\n=== DIRECTIONAL SPLIT per (DOW, TOD), n_events >= 10 ===')
    rows_joint = []
    for (dow, hr), g in cat.groupby(['dow_str', 'tod_hour']):
        n = len(g)
        if n < 10: continue
        n_above = int((g['side']=='above').sum())
        n_below = int((g['side']=='below').sum())
        p_above, lo_a, hi_a = beta_p(n_above, n)
        mean_z = float(g['signed_max_z'].mean())
        rows_joint.append({
            'dow': dow, 'tod_hour': int(hr), 'n_cat_events': n,
            'n_above': n_above, 'n_below': n_below,
            'P_above': round(p_above, 4),
            'cilo': round(lo_a, 4), 'cihi': round(hi_a, 4),
            'mean_signed_z': round(mean_z, 2),
            'mean_abs_z': round(float(g['max_abs_z'].mean()), 2),
        })
    joint_dir_df = pd.DataFrame(rows_joint)
    joint_dir_df.to_csv(os.path.join(args.out_dir, 'dir_skew_by_dow_tod.csv'),
                         index=False)
    # Cells with strong directional skew (CI excludes 0.5)
    strong = joint_dir_df[
        ((joint_dir_df['cihi'] < 0.5) | (joint_dir_df['cilo'] > 0.5))
        & (joint_dir_df['n_cat_events'] >= 15)
    ].copy()
    strong['skew_strength'] = (strong['P_above'] - 0.5).abs()
    strong = strong.sort_values('skew_strength', ascending=False)
    print(f'\nCells with DIRECTIONAL skew (CI excludes 0.5, n>=15):')
    print(strong.head(20).to_string(index=False))
    strong.to_csv(os.path.join(args.out_dir,
                                  'strong_directional_cells.csv'), index=False)

    # ===== Mean-signed-z heatmap: tells us direction of typical cat event =====
    print(f'\n=== MEAN SIGNED MAX_Z per (DOW, TOD), n>=10 ===')
    pivot = joint_dir_df.pivot(index='dow', columns='tod_hour',
                                 values='mean_signed_z').reindex(
        ['Mon','Tue','Wed','Thu','Fri'])
    print(pivot.round(1).to_string())

    # ===== EV CALCULATION =====
    # Position size hypothesis: 1 MNQ contract. Entry at the start of danger window.
    # If cat event triggers: position captures mean_abs_z * SE  in the right direction
    # We don't have SE directly per cat event. Approximate using mean_abs_z * typical_SE.
    # From the 2025_10_29 day stats, avg HL sigma = ~30 pts on MNQ.
    # Conservative: typical SE ~25 pts. So 1σ ≈ 25 pts ≈ $50.
    # mean_abs_z = ~8 means ~200 pts ≈ $400 if captured fully.
    # Of course execution capture is much less. Use a conservative 20% capture factor.
    AVG_SE_PTS = 25.0   # typical 1h SE_close in pts (MNQ baseline)
    CAPTURE_FACTOR = 0.20  # only realize 20% of max excursion (realistic with stops)

    # Combine cat probability (from bar substrate, prior tool) with directional skew
    # First need P(cat) per (DOW, TOD) — we computed this earlier
    bar_subs_csv = 'reports/findings/segments/bayes_table_cat_risk_dow/p_cat_by_dow_x_tod.csv'
    if os.path.exists(bar_subs_csv):
        p_cat_dt = pd.read_csv(bar_subs_csv).rename(
            columns={'P_cat_60m':'P_cat_60m_bar'})
        merged = joint_dir_df.merge(
            p_cat_dt[['dow', 'tod_hour', 'P_cat_60m_bar', 'n']],
            on=['dow', 'tod_hour'], how='left',
            suffixes=('', '_bar'))
        merged['n_bars_sampled'] = merged['n']
        # EV per trade if positioned BEFORE the danger window:
        #   if cat hits (P_cat_60m_bar prob): expected $ if pos same as direction skew
        #   E[$|cat] = mean_abs_z * AVG_SE_PTS * CAPTURE_FACTOR * MNQ_PT_VALUE
        merged['exp_dollar_if_cat'] = (merged['mean_abs_z'] * AVG_SE_PTS
                                          * CAPTURE_FACTOR * MNQ_PT_VALUE).round(0)
        # Direction: take the bias side. If P_above > 0.5, go LONG; else SHORT.
        # If cell has no skew (CI includes 0.5), the position is risky — flag.
        merged['pos_direction'] = np.where(merged['P_above'] > 0.5, 'LONG', 'SHORT')
        # Best-case EV (assumes we always pick the right side):
        merged['EV_directional'] = (merged['P_cat_60m_bar'].fillna(0)
                                       * merged['exp_dollar_if_cat']).round(0)
        # STRADDLE EV: capture move regardless of direction (less but more reliable)
        # Assume both legs cost half the directional $ each, only one wins
        merged['EV_straddle'] = (merged['P_cat_60m_bar'].fillna(0)
                                    * merged['exp_dollar_if_cat'] * 0.5).round(0)

        print(f'\n=== TOP 20 EV (directional) CELLS — Tuesday/Wednesday UTC 1 zone ===')
        m_show = merged[merged['n_cat_events'] >= 15].copy()
        m_show['abs_skew'] = (m_show['P_above'] - 0.5).abs()
        top_ev = m_show.sort_values('EV_directional', ascending=False).head(20)
        cols = ['dow', 'tod_hour', 'n_cat_events', 'P_above', 'mean_signed_z',
                 'mean_abs_z', 'P_cat_60m_bar', 'exp_dollar_if_cat',
                 'pos_direction', 'EV_directional', 'EV_straddle']
        print(top_ev[cols].to_string(index=False))
        merged.to_csv(os.path.join(args.out_dir, 'cat_harvest_ev_table.csv'),
                       index=False)

    # ===== CHARTS =====
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. DOW directional skew (top-left)
    ax = axes[0][0]
    sub = dir_df.set_index('dow')
    dow_order = [d for d in ['Mon','Tue','Wed','Thu','Fri'] if d in sub.index]
    sub = sub.reindex(dow_order)
    x = np.arange(len(dow_order))
    p = sub['P_above'].values
    ax.bar(x, p * 100, color=['#43A047' if pi > 0.5 else '#E53935' for pi in p],
             alpha=0.85)
    ax.axhline(50, color='black', ls='--', lw=0.8, label='neutral 50%')
    ax.errorbar(x, sub['P_above'].values * 100,
                  yerr=[(sub['P_above']-sub['cilo']).values * 100,
                         (sub['cihi']-sub['P_above']).values * 100],
                  fmt='none', ecolor='black', capsize=4)
    for xi, pi in zip(x, p): ax.text(xi, pi*100+1, f'{pi*100:.1f}%', ha='center')
    ax.set_xticks(x); ax.set_xticklabels(dow_order)
    ax.set_ylabel('P(above_high) %')
    ax.set_title(f'Cat-event direction skew per DOW  (|max_z|>={args.cat_threshold})\n'
                  f'green=rally bias, red=crash bias')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. DOW × TOD mean signed z heatmap (top-right)
    ax = axes[0][1]
    pivot_signed = joint_dir_df.pivot(index='dow', columns='tod_hour',
                                        values='mean_signed_z').reindex(
        ['Mon','Tue','Wed','Thu','Fri'])
    if not pivot_signed.empty:
        vabs = float(np.nanmax(np.abs(pivot_signed.values)))
        im = ax.imshow(pivot_signed.values, cmap='RdYlGn', aspect='auto',
                        vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(len(pivot_signed.columns)))
        ax.set_xticklabels(pivot_signed.columns, fontsize=8)
        ax.set_yticks(range(len(pivot_signed.index)))
        ax.set_yticklabels(pivot_signed.index)
        ax.set_xlabel('UTC hour'); ax.set_ylabel('DOW')
        ax.set_title('Mean signed max_z per (DOW, TOD) — green=rally bias, red=crash')
        plt.colorbar(im, ax=ax, label='mean signed_max_z')
        for i in range(pivot_signed.shape[0]):
            for j in range(pivot_signed.shape[1]):
                v = pivot_signed.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                              fontsize=6)

    # 3. EV directional heatmap (bottom-left)
    if os.path.exists(bar_subs_csv):
        ax = axes[1][0]
        pivot_ev = merged.pivot(index='dow', columns='tod_hour',
                                  values='EV_directional').reindex(
            ['Mon','Tue','Wed','Thu','Fri'])
        if not pivot_ev.empty:
            im = ax.imshow(pivot_ev.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(pivot_ev.columns)))
            ax.set_xticklabels(pivot_ev.columns, fontsize=8)
            ax.set_yticks(range(len(pivot_ev.index)))
            ax.set_yticklabels(pivot_ev.index)
            ax.set_xlabel('UTC hour'); ax.set_ylabel('DOW')
            ax.set_title(f'EV directional ($ per bar in this cell)\n'
                          f'P_cat * E[|move|] * capture * pt_value')
            plt.colorbar(im, ax=ax, label='EV $')
            for i in range(pivot_ev.shape[0]):
                for j in range(pivot_ev.shape[1]):
                    v = pivot_ev.values[i, j]
                    if not np.isnan(v) and abs(v) >= 10:
                        ax.text(j, i, f'${v:.0f}', ha='center', va='center',
                                  fontsize=5)

    # 4. EV straddle heatmap (bottom-right)
    if os.path.exists(bar_subs_csv):
        ax = axes[1][1]
        pivot_str = merged.pivot(index='dow', columns='tod_hour',
                                   values='EV_straddle').reindex(
            ['Mon','Tue','Wed','Thu','Fri'])
        if not pivot_str.empty:
            im = ax.imshow(pivot_str.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(pivot_str.columns)))
            ax.set_xticklabels(pivot_str.columns, fontsize=8)
            ax.set_yticks(range(len(pivot_str.index)))
            ax.set_yticklabels(pivot_str.index)
            ax.set_xlabel('UTC hour'); ax.set_ylabel('DOW')
            ax.set_title(f'EV straddle ($ per bar; both legs split capture)')
            plt.colorbar(im, ax=ax, label='EV $')
            for i in range(pivot_str.shape[0]):
                for j in range(pivot_str.shape[1]):
                    v = pivot_str.values[i, j]
                    if not np.isnan(v) and abs(v) >= 5:
                        ax.text(j, i, f'${v:.0f}', ha='center', va='center',
                                  fontsize=5)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'cat_harvest_analysis.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()

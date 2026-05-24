"""At-bar Bayesian risk: P(catastrophic event in next 60min | t, state).

For ANY 5s bar in the dataset, computes whether a catastrophic event
(|max_z| >= cat_threshold) starts within the next look_ahead_min minutes.
Then aggregates by conditioning axes:
    TOD hour, day-of-week, current sigma_rank bucket, current slope_q

Produces the 'in-flight risk' table — when sitting at any bar, the
conditional probability of facing a tail event soon.

This complements the per-event magnitude/duration tables. Those answer
'given an event triggered, how big and how long?'. This one answers
'at this bar, am I about to be in trouble?'.

USAGE
    python tools/bayes_table_at_bar_cat_risk.py
    python tools/bayes_table_at_bar_cat_risk.py --cat-threshold 8.0 --look-ahead-min 60
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.event_bucket_15m_crm import _compute_day_crm_features
from tools.segment_day_motif_melody import _load_5s


def beta_p(k, n):
    a, b = k+1, n-k+1
    return float(a/(a+b)), float(beta_dist.ppf(0.025, a, b)), float(beta_dist.ppf(0.975, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cat-threshold', type=float, default=8.0,
                     help='|max_z| threshold defining a catastrophic event')
    ap.add_argument('--look-ahead-min', type=int, default=60,
                     help='Forward-window in minutes')
    ap.add_argument('--sample-every-5s-bars', type=int, default=12,
                     help='Subsample stride (12 = every 60s)')
    ap.add_argument('--macro-csv',
                     default='reports/findings/band_touch_aggregation/macro_events_1h_hl.csv')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_at_bar_cat_risk')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all events, filter to catastrophic
    evs = pd.read_csv(args.macro_csv)
    cat = evs[evs['max_abs_z'] >= args.cat_threshold].copy()
    print(f'Catastrophic events (|max_z|>={args.cat_threshold}): {len(cat)} on {cat["day"].nunique()} days')
    print(f'  IS: {(cat["split"]=="IS").sum()}   OOS: {(cat["split"]=="OOS").sum()}')

    look_ahead_s = args.look_ahead_min * 60

    # Bucket edges from event substrate (for sigma_rank and slope conditioning)
    sub = pd.read_parquet('reports/findings/segments/bayes_table_v0_location/event_substrate.parquet')
    is_pop = sub[sub['split'] == 'IS']
    quant_edges = {}
    for col, n_bins in [('slope', 5), ('sigma_rank', 5)]:
        v = is_pop[col].dropna().values
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        edges[0] = -np.inf; edges[-1] = np.inf
        quant_edges[col] = edges

    def bucket(value, col):
        if not np.isfinite(value):
            return -1
        edges = quant_edges[col]
        return int(np.clip(np.searchsorted(edges, value, side='right') - 1,
                            0, len(edges) - 2))

    # For every (day, sample-bar): is there a cat event within next look_ahead_min?
    # Process per day to keep memory tractable
    all_days = sorted(evs['day'].unique())
    rows = []
    cat_by_day = {d: cat[cat['day']==d][['start_ts','max_abs_z']].values
                   for d in cat['day'].unique()}
    split_by_day = evs.groupby('day')['split'].first().to_dict()

    for day in tqdm(all_days, desc='days'):
        df_5s = _load_5s(day)
        if df_5s.empty: continue
        ts = df_5s['timestamp'].values.astype(np.int64)
        feats = _compute_day_crm_features(day)
        if feats is None: continue
        # Sample stride
        sample_idx = np.arange(0, len(ts), args.sample_every_5s_bars)
        cat_starts = cat_by_day.get(day, np.zeros((0, 2)))
        cat_start_ts = cat_starts[:, 0].astype(np.int64) if len(cat_starts) else np.array([], dtype=np.int64)
        for i in sample_idx:
            t = int(ts[i])
            # Is there a cat event starting in (t, t+look_ahead_s]?
            if len(cat_start_ts) > 0:
                upcoming = (cat_start_ts > t) & (cat_start_ts <= t + look_ahead_s)
                cat_in_window = bool(upcoming.any())
            else:
                cat_in_window = False
            # At-bar features
            slope = feats['slope'][i] if i < len(feats['slope']) else np.nan
            sigma_rank = feats['sigma_rank'][i] if i < len(feats['sigma_rank']) else np.nan
            tod_hour = datetime.fromtimestamp(t, tz=timezone.utc).hour
            dow = datetime.fromtimestamp(t, tz=timezone.utc).strftime('%a')
            # Has a cat event ALREADY happened today (before this bar)?
            cat_already_today = bool(((cat_start_ts > 0) & (cat_start_ts <= t)).any())
            rows.append({
                'day': day,
                'split': split_by_day.get(day, 'NA'),
                'ts': t,
                'tod_hour': tod_hour,
                'dow': dow,
                'slope_q': bucket(slope, 'slope'),
                'sigma_rank_q': bucket(sigma_rank, 'sigma_rank'),
                'cat_already_today': cat_already_today,
                'cat_in_next_60m': cat_in_window,
            })
    bar_df = pd.DataFrame(rows)
    print(f'\nSampled {len(bar_df):,} bars across {bar_df["day"].nunique()} days')

    bar_df.to_parquet(os.path.join(args.out_dir, 'bar_substrate.parquet'), index=False)

    # ===== Aggregations =====
    print(f'\n=== UNCONDITIONAL P(cat event in next {args.look_ahead_min}min): {100*bar_df["cat_in_next_60m"].mean():.2f}% ===')

    # By TOD hour
    print(f'\n=== P(cat in next {args.look_ahead_min}min | TOD hour) ===')
    rows_tod = []
    for hr, g in bar_df.groupby('tod_hour'):
        n = len(g)
        k = int(g['cat_in_next_60m'].sum())
        p, lo, hi = beta_p(k, n)
        rows_tod.append({'tod_hour': hr, 'n': n, 'P_cat_60m': round(p, 4),
                          'cilo': round(lo, 4), 'cihi': round(hi, 4)})
    tod_df = pd.DataFrame(rows_tod).sort_values('tod_hour')
    tod_df.to_csv(os.path.join(args.out_dir, 'p_cat_by_tod.csv'), index=False)
    print(tod_df.to_string(index=False))

    # By sigma_rank_q
    print(f'\n=== P(cat in next {args.look_ahead_min}min | sigma_rank_q) ===')
    rows_sig = []
    for sq, g in bar_df.groupby('sigma_rank_q'):
        if sq < 0: continue
        n = len(g); k = int(g['cat_in_next_60m'].sum())
        p, lo, hi = beta_p(k, n)
        rows_sig.append({'sigma_rank_q': int(sq), 'n': n,
                          'P_cat_60m': round(p, 4),
                          'cilo': round(lo, 4), 'cihi': round(hi, 4)})
    sig_df = pd.DataFrame(rows_sig).sort_values('sigma_rank_q')
    sig_df.to_csv(os.path.join(args.out_dir, 'p_cat_by_sigma.csv'), index=False)
    print(sig_df.to_string(index=False))

    # By cat_already_today
    print(f'\n=== P(cat in next {args.look_ahead_min}min | cat_already_today) ===')
    rows_already = []
    for already, g in bar_df.groupby('cat_already_today'):
        n = len(g); k = int(g['cat_in_next_60m'].sum())
        p, lo, hi = beta_p(k, n)
        rows_already.append({'cat_already_today': bool(already), 'n': n,
                              'P_cat_60m': round(p, 4),
                              'cilo': round(lo, 4), 'cihi': round(hi, 4)})
    already_df = pd.DataFrame(rows_already)
    already_df.to_csv(os.path.join(args.out_dir, 'p_cat_by_already.csv'), index=False)
    print(already_df.to_string(index=False))

    # JOINT: TOD x sigma_rank_q
    print(f'\n=== JOINT P(cat in next {args.look_ahead_min}min | TOD, sigma_q) ===')
    rows_j = []
    for (hr, sq), g in bar_df.groupby(['tod_hour', 'sigma_rank_q']):
        if sq < 0: continue
        n = len(g)
        if n < 50: continue
        k = int(g['cat_in_next_60m'].sum())
        p, lo, hi = beta_p(k, n)
        rows_j.append({'tod_hour': hr, 'sigma_rank_q': int(sq), 'n': n,
                        'P_cat_60m': round(p, 4),
                        'cilo': round(lo, 4), 'cihi': round(hi, 4)})
    joint_df = pd.DataFrame(rows_j)
    joint_df.to_csv(os.path.join(args.out_dir, 'p_cat_by_tod_x_sigma.csv'), index=False)
    # Show top 20 highest P(cat) cells
    print(f'\nTop 20 highest P(cat in next 60m) cells (n>=50):')
    print(joint_df.sort_values('P_cat_60m', ascending=False).head(20).to_string(index=False))
    print(f'\nBottom 10 lowest (safest) cells:')
    print(joint_df.sort_values('P_cat_60m').head(10).to_string(index=False))

    # ===== CHARTS =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    # TOD bar chart
    ax = axes[0][0]
    ax.bar(tod_df['tod_hour'], tod_df['P_cat_60m'] * 100,
            color='#E53935', alpha=0.8)
    base = bar_df['cat_in_next_60m'].mean() * 100
    ax.axhline(base, color='black', ls='--', lw=0.8, label=f'baseline {base:.2f}%')
    ax.set_xlabel('UTC hour'); ax.set_ylabel('P(cat in next 60m) %')
    ax.set_title('In-flight risk by TOD hour'); ax.legend()
    ax.set_xticks(range(0, 24, 2)); ax.grid(True, alpha=0.3)

    # Sigma bar chart
    ax = axes[0][1]
    ax.bar(sig_df['sigma_rank_q'], sig_df['P_cat_60m'] * 100,
            color='#FB8C00', alpha=0.8)
    ax.axhline(base, color='black', ls='--', lw=0.8, label=f'baseline {base:.2f}%')
    ax.set_xlabel('sigma_rank quintile'); ax.set_ylabel('P(cat in next 60m) %')
    ax.set_title('In-flight risk by current sigma_rank'); ax.legend()
    ax.grid(True, alpha=0.3)

    # Already-today comparison
    ax = axes[1][0]
    bars = ax.bar(['no cat yet today', 'cat already today'],
                    already_df['P_cat_60m'].values * 100,
                    color=['#43A047', '#E53935'], alpha=0.85)
    ax.axhline(base, color='black', ls='--', lw=0.8, label=f'baseline {base:.2f}%')
    ax.set_ylabel('P(cat in next 60m) %')
    ax.set_title('In-flight risk by prior-cat-today (clustering test)')
    ax.legend(); ax.grid(True, alpha=0.3)
    for bar, p in zip(bars, already_df['P_cat_60m'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  f'{100*p:.2f}%', ha='center', fontsize=10)

    # Joint heatmap
    ax = axes[1][1]
    pivot = joint_df.pivot(index='sigma_rank_q', columns='tod_hour',
                             values='P_cat_60m')
    if not pivot.empty:
        im = ax.imshow(pivot.values * 100, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'sigma_q{q}' for q in pivot.index])
        ax.set_xlabel('UTC hour'); ax.set_ylabel('sigma_rank_q')
        ax.set_title('P(cat in next 60m) heatmap: TOD x sigma_q')
        plt.colorbar(im, ax=ax, label='%')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{100*v:.0f}', ha='center', va='center',
                              fontsize=6, color='black')

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, f'p_cat_in_next_{args.look_ahead_min}m.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()

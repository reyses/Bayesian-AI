"""V0 Bayesian Table — trade-location-keyed (3-anchor schema).

COMPLEMENT to bayes_table_v0_joint_chord.py (NOT a replacement).
The chord-shape table answers "what shape are we in".
THIS table answers "where is the price relative to the 3-anchor envelope".

POPULATION
    6,495 macro events at 1h HL k>=3 sigma (>=60s duration), pre-aggregated by
    `tools/band_touch_aggregation.py` and saved to
    `reports/findings/band_touch_aggregation/macro_events_1h_hl.csv`.
    These are the high-conviction TRADE LOCATIONS in the 3-anchor schema:
    price has pushed past the 1h M_high (rally) or M_low (crash) by k=3 sigma
    and held there for at least 60 seconds.

CONDITIONING AXES (15m CRM context, computed at event entry, NO LOOKAHEAD)
    side          above/below       (which side of the 1h envelope)
    anchor        high/low          (which 1h regression mean)
    slope_q       5 bins            (1h M_close 60-min slope)
    curv_q        3 bins            (1h M_close 60-min curvature)
    z_close_q     5 bins            (5s close vs 1h M_close)
    sigma_rank_q  5 bins            (1h SE_close 60-min percentile)
    r2adj_q       5 bins            (5s close 5-min linear-fit R^2_adj)
    tod_hour      0-23
    dow           Mon-Fri

OUTCOMES (computed forward from entry, NO LOOKAHEAD into outcomes for
conditioning — features are at-entry only)
    P_continue_60m    fraction where event still active 60min after entry
    P_continue_30m    same at 30min
    duration_min      mean event run length
    max_abs_z         mean peak deviation
    fwd_ret_60m       mean 5s close return 60min after entry
    fwd_mfe_60m       mean max favorable excursion in window
    fwd_mae_60m       mean max adverse excursion in window

OUTPUT
    1. MARGINAL TABLES — per axis, P(outcome | axis_q) with IS/OOS split
       and sign-match flag. Tells us which axis carries stable signal.
    2. JOINT TABLE — top-3-axes joint cell (side, slope_q, z_close_q)
       with hierarchical pooling for sparse cells.
    3. SIGN-STABILITY REPORT — which (axis, bin, outcome) cells survive IS->OOS.
    4. CHARTS — marginal P_continue heatmaps + joint cell map.

USAGE
    python tools/bayes_table_v0_trade_location.py
    python tools/bayes_table_v0_trade_location.py --min-cell-n 10
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.event_bucket_15m_crm import (
    _compute_day_crm_features as _build_day_features,
    _feature_at_event, _bucket_quantile, AXIS_BIN_DEFS,
)
from tools.segment_day_motif_melody import _load_5s


HORIZONS_FWD_S = {
    'h5m':  300,
    'h15m': 900,
    'h30m': 1800,
    'h60m': 3600,
}

OUTCOME_COLS = ['P_continue_30m', 'P_continue_60m',
                 'fwd_ret_60m', 'duration_min', 'max_abs_z']


def beta_posterior(k: int, n: int) -> tuple[float, float, float]:
    a, b = k + 1, n - k + 1
    return (float(a / (a + b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def attach_features_and_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    """For each event row, compute (a) 15m CRM features at entry, (b) forward
    outcomes from entry. Caches per-day arrays so each day loads once."""
    feats_by_day = {}
    fwd_close_by_day = {}
    rows = []
    for day, sub in tqdm(events.groupby('day'), desc='per-day features'):
        if day not in feats_by_day:
            feats_by_day[day] = _build_day_features(day)
            df_5s = _load_5s(day)
            if df_5s.empty:
                fwd_close_by_day[day] = None
            else:
                fwd_close_by_day[day] = (
                    df_5s['timestamp'].values.astype(np.int64),
                    df_5s['close'].values.astype(np.float64))
        day_feats = feats_by_day[day]
        ts_close = fwd_close_by_day[day]
        for _, e in sub.iterrows():
            f = _feature_at_event(day_feats, int(e['start_ts']))
            r = e.to_dict()
            if f is None:
                r.update({'slope': np.nan, 'curv': np.nan,
                           'z_close_at_entry': np.nan,
                           'sigma_rank': np.nan, 'r2adj_5m': np.nan})
            else:
                r.update({
                    'slope':            f['slope'],
                    'curv':             f['curv'],
                    'z_close_at_entry': f['z_close'],
                    'sigma_rank':       f['sigma_rank'],
                    'r2adj_5m':         f['r2adj_5m'],
                })
            # Forward outcomes from entry timestamp
            if ts_close is None:
                for hk in HORIZONS_FWD_S:
                    r[f'fwd_ret_{hk}'] = np.nan
                    r[f'fwd_mfe_{hk}'] = np.nan
                    r[f'fwd_mae_{hk}'] = np.nan
            else:
                ts, close = ts_close
                start = int(e['start_ts'])
                i_start = int(np.searchsorted(ts, start))
                if i_start >= len(close):
                    base = np.nan
                else:
                    base = close[i_start]
                for hk, h_s in HORIZONS_FWD_S.items():
                    i_end = int(np.searchsorted(ts, start + h_s))
                    if (i_end >= len(close)) or (not np.isfinite(base)):
                        r[f'fwd_ret_{hk}'] = np.nan
                        r[f'fwd_mfe_{hk}'] = np.nan
                        r[f'fwd_mae_{hk}'] = np.nan
                    else:
                        r[f'fwd_ret_{hk}'] = float(close[i_end] - base)
                        w = close[i_start:i_end + 1]
                        r[f'fwd_mfe_{hk}'] = float(w.max() - base)
                        r[f'fwd_mae_{hk}'] = float(w.min() - base)
            rows.append(r)
    return pd.DataFrame(rows)


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add q-bucket columns based on event-population quantiles."""
    df = df.copy()
    for axis, defs in AXIS_BIN_DEFS.items():
        col = {'slope': 'slope', 'curvature': 'curv',
               'z_close': 'z_close_at_entry', 'sigma_rank': 'sigma_rank',
               'r2adj_5m': 'r2adj_5m'}[axis]
        df[f'{axis}_q'] = _bucket_quantile(df[col].values, defs['n_bins'])
    return df


def add_continuation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """An event 'continues' at horizon h if duration_s + start_ts >= start_ts + h.
    i.e. duration_s >= h."""
    df = df.copy()
    df['P_continue_30m'] = (df['duration_s'] >= 1800).astype(int)
    df['P_continue_60m'] = (df['duration_s'] >= 3600).astype(int)
    return df


def marginal_table(events: pd.DataFrame, axis: str,
                   group_cols: list[str],
                   min_cell_n: int = 10) -> pd.DataFrame:
    """For each (split, *group_cols, axis_q) compute outcome means + Beta CI."""
    rows = []
    qcol = f'{axis}_q'
    for split in ['IS', 'OOS']:
        sub = events[events['split'] == split]
        for keys, g in sub.groupby(['side', 'anchor', qcol]):
            n = len(g)
            if n < min_cell_n:
                continue
            side, anchor, q = keys
            if q < 0:
                continue
            rec = {'split': split, 'side': side, 'anchor': anchor,
                    'axis': axis, 'bin': int(q), 'n': n}
            # P_continue_60m and 30m as Beta posteriors
            for pcol in ['P_continue_30m', 'P_continue_60m']:
                k = int(g[pcol].sum())
                p, lo, hi = beta_posterior(k, n)
                rec[f'{pcol}_mean'] = round(p, 4)
                rec[f'{pcol}_cilo'] = round(lo, 4)
                rec[f'{pcol}_cihi'] = round(hi, 4)
            # continuous outcomes
            for col in ['fwd_ret_h60m', 'fwd_mfe_h60m', 'fwd_mae_h60m',
                         'duration_min', 'max_abs_z']:
                if col in g.columns:
                    rec[f'{col}_mean'] = round(float(g[col].mean()), 3)
            rows.append(rec)
    return pd.DataFrame(rows)


def sign_match_axis(marg_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """For each (side, anchor, axis, bin), compare IS vs OOS sign of (mean - 0.5)
    for P_*  outcomes, or sign of mean for fwd_ret outcomes."""
    rows = []
    base = outcome
    is_prob = outcome.startswith('P_')
    pivot = marg_df.pivot_table(
        index=['side', 'anchor', 'axis', 'bin'],
        columns='split',
        values=f'{base}_mean' if is_prob else f'{base}_mean',
        aggfunc='first',
    ).reset_index()
    if 'IS' not in pivot.columns or 'OOS' not in pivot.columns:
        return pd.DataFrame()
    for _, r in pivot.iterrows():
        is_v = r['IS']
        oo_v = r['OOS']
        if pd.isna(is_v) or pd.isna(oo_v):
            sm = -1
        elif is_prob:
            sm = int(np.sign(is_v - 0.5) == np.sign(oo_v - 0.5))
        else:
            sm = int(np.sign(is_v) == np.sign(oo_v))
        rows.append({
            'side': r['side'], 'anchor': r['anchor'], 'axis': r['axis'],
            'bin': int(r['bin']),
            'IS': round(float(is_v), 4) if pd.notna(is_v) else np.nan,
            'OOS': round(float(oo_v), 4) if pd.notna(oo_v) else np.nan,
            'sign_match': sm,
        })
    return pd.DataFrame(rows)


def joint_table(events: pd.DataFrame, axes: list[str],
                 min_cell_n: int = 10) -> pd.DataFrame:
    """Joint cell P(continuation_60m) over (side, anchor, *axes_q)."""
    rows = []
    qcols = [f'{a}_q' for a in axes]
    for split in ['IS', 'OOS']:
        sub = events[events['split'] == split]
        for keys, g in sub.groupby(['side', 'anchor'] + qcols):
            n = len(g)
            if n < min_cell_n:
                continue
            side, anchor, *qs = keys
            if any(q < 0 for q in qs):
                continue
            k = int(g['P_continue_60m'].sum())
            p, lo, hi = beta_posterior(k, n)
            rec = {
                'split': split, 'side': side, 'anchor': anchor,
                'n': n,
                'P_continue_60m_mean': round(p, 4),
                'cilo': round(lo, 4), 'cihi': round(hi, 4),
                'fwd_ret_60m_mean': round(float(g['fwd_ret_h60m'].mean()), 3)
                                       if 'fwd_ret_h60m' in g.columns else np.nan,
                'duration_min_mean': round(float(g['duration_min'].mean()), 2),
                'max_abs_z_mean': round(float(g['max_abs_z'].mean()), 2),
            }
            for ax, q in zip(axes, qs):
                rec[f'{ax}_q'] = int(q)
            rows.append(rec)
    return pd.DataFrame(rows)


def render_marginal_chart(marg_df: pd.DataFrame, out_dir: str):
    """One subplot per (axis × outcome) showing IS and OOS bin means."""
    if marg_df.empty:
        return
    axes_list = marg_df['axis'].unique().tolist()
    outcomes = ['P_continue_30m', 'P_continue_60m']
    fig, axes = plt.subplots(len(outcomes), len(axes_list),
                              figsize=(4 * len(axes_list), 3.5 * len(outcomes)),
                              squeeze=False)
    for i, oc in enumerate(outcomes):
        for j, ax_name in enumerate(axes_list):
            ax = axes[i][j]
            sub = marg_df[marg_df['axis'] == ax_name]
            for split, color in [('IS', '#1E88E5'), ('OOS', '#FB8C00')]:
                ssub = sub[sub['split'] == split]
                if ssub.empty:
                    continue
                # Aggregate across (side, anchor) by mean for chart simplicity
                agg = ssub.groupby('bin').agg(
                    p=(f'{oc}_mean', 'mean'),
                    lo=(f'{oc}_cilo', 'mean'),
                    hi=(f'{oc}_cihi', 'mean'),
                    n=('n', 'sum'),
                ).reset_index()
                ax.errorbar(agg['bin'], agg['p'],
                              yerr=[agg['p'] - agg['lo'], agg['hi'] - agg['p']],
                              fmt='o-', color=color, label=f'{split} (n_total={agg["n"].sum()})',
                              capsize=3, alpha=0.85)
            ax.axhline(0.5, color='gray', lw=0.6, alpha=0.5)
            ax.set_title(f'{oc} | {ax_name}', fontsize=10)
            ax.set_xlabel('bin'); ax.set_ylabel('P')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Trade-location Bayesian table V0 — marginal P(continuation) per axis',
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, 'v0_marginals.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  marginal chart -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro-csv',
                     default='reports/findings/band_touch_aggregation/macro_events_1h_hl.csv')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_v0_location')
    ap.add_argument('--min-cell-n', type=int, default=10)
    ap.add_argument('--min-joint-n', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading {args.macro_csv}...')
    events = pd.read_csv(args.macro_csv)
    events['duration_min'] = events['duration_s'] / 60.0
    print(f'Loaded {len(events):,} macro events  '
          f'(IS: {(events["split"]=="IS").sum():,}  OOS: {(events["split"]=="OOS").sum():,})')

    # Attach 15m CRM features at entry + forward returns
    events = attach_features_and_outcomes(events)
    events = add_continuation_flags(events)
    events = add_buckets(events)

    # Save the per-event substrate
    sub_path = os.path.join(args.out_dir, 'event_substrate.parquet')
    events.to_parquet(sub_path, index=False)
    print(f'\nPer-event substrate -> {sub_path}')
    print(f'  n_events: {len(events):,}')
    print(f'  n with all 5 features: {events[["slope","curv","z_close_at_entry","sigma_rank","r2adj_5m"]].notna().all(axis=1).sum():,}')

    # ========== MARGINAL TABLES per axis ==========
    print(f'\n=== MARGINAL TABLES per axis (min_cell_n={args.min_cell_n}) ===')
    all_marg = []
    for axis in AXIS_BIN_DEFS:
        m = marginal_table(events, axis, [], args.min_cell_n)
        if m.empty:
            print(f'  {axis}: no cells'); continue
        all_marg.append(m)
        m.to_csv(os.path.join(args.out_dir, f'marginal_{axis}.csv'), index=False)
        # Sign-match
        sm = sign_match_axis(m, 'P_continue_60m')
        sm.to_csv(os.path.join(args.out_dir, f'sign_match_{axis}_P_continue_60m.csv'),
                   index=False)
        match_rate = sm['sign_match'].eq(1).mean() if not sm.empty else 0
        print(f'  {axis}: {len(m)} (split,side,anchor,bin) cells   '
              f'sign-match rate IS-vs-OOS: {100*match_rate:.0f}%')
    if all_marg:
        all_marg_df = pd.concat(all_marg)
        render_marginal_chart(all_marg_df, args.out_dir)

    # ========== JOINT TABLE: side × anchor × slope × z_close × sigma_rank ==========
    print(f'\n=== JOINT TABLE: side × anchor × slope × z_close × sigma_rank ===')
    j = joint_table(events, ['slope', 'z_close', 'sigma_rank'],
                     min_cell_n=args.min_joint_n)
    j.to_csv(os.path.join(args.out_dir, 'joint_5axis.csv'), index=False)
    print(f'  joint cells: {len(j)}')

    # ========== SUMMARY: top-edge marginals + sign-stable axes ==========
    summary_rows = []
    for axis in AXIS_BIN_DEFS:
        sm_path = os.path.join(args.out_dir, f'sign_match_{axis}_P_continue_60m.csv')
        if not os.path.exists(sm_path):
            continue
        sm = pd.read_csv(sm_path)
        if sm.empty:
            continue
        # Filter to cells where both IS and OOS have data
        valid = sm[sm['sign_match'].isin([0, 1])]
        if valid.empty:
            continue
        # |IS - 0.5| edge weighted by sign-match
        valid = valid.copy()
        valid['IS_edge'] = (valid['IS'] - 0.5).abs()
        sign_stable_edge = valid[valid['sign_match'] == 1]['IS_edge'].mean()
        summary_rows.append({
            'axis': axis,
            'n_cells_with_oos': len(valid),
            'sign_stable_pct': round(100 * valid['sign_match'].mean(), 1),
            'mean_edge_when_stable': round(sign_stable_edge, 4),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(
        'sign_stable_pct', ascending=False)
    summary_df.to_csv(os.path.join(args.out_dir, 'axis_stability_summary.csv'),
                       index=False)
    print(f'\n=== AXIS STABILITY SUMMARY ===')
    print(summary_df.to_string(index=False))

    # Print top high-edge OOS-confirmed cells for each axis
    print(f'\n=== TOP HIGH-EDGE OOS-CONFIRMED CELLS PER AXIS ===')
    for axis in AXIS_BIN_DEFS:
        sm_path = os.path.join(args.out_dir, f'sign_match_{axis}_P_continue_60m.csv')
        if not os.path.exists(sm_path):
            continue
        sm = pd.read_csv(sm_path)
        if sm.empty:
            continue
        sm = sm[sm['sign_match'] == 1].copy()
        if sm.empty:
            continue
        sm['IS_edge'] = (sm['IS'] - 0.5).abs()
        top = sm.sort_values('IS_edge', ascending=False).head(3)
        print(f'\n  {axis}:')
        for _, r in top.iterrows():
            print(f'    side={r["side"]:<5s}  anchor={r["anchor"]:<5s}  '
                   f'bin={int(r["bin"])}   IS={r["IS"]:.3f}  OOS={r["OOS"]:.3f}')


if __name__ == '__main__':
    main()

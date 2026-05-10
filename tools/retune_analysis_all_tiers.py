"""Run the validated E1 (z-band) + E2 (VETO) analysis on every NMP-based tier.

Finds the optimal |z_at_entry| band and 1h regime VETO cells per tier.

Produces:
    reports/findings/segments/retune_per_tier/per_tier_retune.csv
    reports/findings/segments/retune_per_tier/per_tier_retune.md

USAGE
    python tools/retune_analysis_all_tiers.py
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Only NMP-based tiers (have z_se in extras)
NMP_TIERS = ['FADE_CALM', 'FADE_MOMENTUM', 'FADE_AGAINST',
             'RIDE_CALM',  'RIDE_MOMENTUM', 'RIDE_AGAINST',
             'NMP_FADE_RAW','NMP_RIDE_RAW']


def pf(g):
    w = g[g.pnl>0].pnl.sum()
    l = abs(g[g.pnl<0].pnl.sum())
    return (w/l - 1) if l > 0 else float('inf')


def load_tier_trades(tier: str) -> pd.DataFrame:
    rows = []
    for split in ('IS','OOS'):
        p = f'training_iso_v2/output/{split.lower()}_{tier}.pkl'
        if not os.path.exists(p): continue
        with open(p,'rb') as f:
            trades = pickle.load(f)
        for t in trades:
            z = t.extras.get('z_se') if isinstance(t.extras, dict) else None
            if z is None or pd.isna(z): continue
            rows.append({
                'tier': tier, 'split': split, 'pnl': float(t.pnl),
                'direction': t.direction, 'z_abs': abs(float(z)),
                'z_signed': float(z), 'entry_ts': int(t.entry_ts),
                'entry_day': t.entry_day,
            })
    return pd.DataFrame(rows)


def attach_1h_z(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    by_day = defaultdict(list)
    for idx, r in df.iterrows():
        by_day[r['entry_day']].append((idx, r['entry_ts']))
    z_1h_arr = np.full(len(df), np.nan)
    for day, items in by_day.items():
        path = f'DATA/ATLAS/FEATURES_5s_v2/L3_1h/{day}.parquet'
        if not os.path.exists(path): continue
        df_v2 = pd.read_parquet(path)
        ts_a = df_v2['timestamp'].astype(np.int64).values
        z_a  = df_v2['L3_1h_z_se_12'].values
        for idx, ts in items:
            i = np.searchsorted(ts_a, ts, side='right') - 1
            if 0 <= i < len(ts_a) and not pd.isna(z_a[i]):
                z_1h_arr[idx] = float(z_a[i])
    df['z_1h'] = z_1h_arr
    return df


def cell_summary(g: pd.DataFrame) -> dict:
    if g.empty: return dict(n=0, total=0, mean=0, is_t=0, oos_t=0, pf_wr=0)
    is_t = g[g.split=='IS'].pnl.sum()
    oos_t = g[g.split=='OOS'].pnl.sum()
    return dict(n=len(g),
                  total=round(g.pnl.sum(),2),
                  mean=round(g.pnl.mean(),3),
                  is_t=round(is_t,2), oos_t=round(oos_t,2),
                  pf_wr=round(pf(g),3))


def find_optimal_band(df: pd.DataFrame) -> tuple:
    """Search common bands; pick the one with highest PF_WR that has both
    IS and OOS positive."""
    candidates = [
        (1.0, 3.0), (1.0, 2.5), (1.0, 2.2),
        (1.2, 2.5), (1.2, 2.2), (1.2, 2.0),
        (1.5, 3.0), (1.5, 2.5), (1.5, 2.2), (1.5, 2.0), (1.5, 1.8),
        (1.8, 3.0), (1.8, 2.5), (1.8, 2.2), (1.8, 2.0),
    ]
    best = None
    for lo, hi in candidates:
        sub = df[(df.z_abs >= lo) & (df.z_abs <= hi)]
        if len(sub) < 100: continue
        s = cell_summary(sub)
        if s['is_t'] <= 0 or s['oos_t'] <= 0: continue
        if best is None or s['pf_wr'] > best[2]['pf_wr']:
            best = (lo, hi, s)
    return best


def find_veto_cells(df: pd.DataFrame) -> list:
    """Return list of (direction, 1h_category) cells where BOTH IS and OOS
    are negative (structural losers)."""
    df = df.copy()
    cats = []
    for _, r in df.iterrows():
        if r['direction'] == 'short':
            cats.append('aligned' if r['z_1h'] >= +0.3 else 'opposed' if r['z_1h'] <= -0.3 else 'neutral')
        else:
            cats.append('aligned' if r['z_1h'] <= -0.3 else 'opposed' if r['z_1h'] >= +0.3 else 'neutral')
    df['cat'] = cats
    veto = []
    for direction in ('long','short'):
        for cat in ('aligned','opposed','neutral'):
            g = df[(df.direction==direction) & (df.cat==cat)]
            if len(g) < 50: continue
            s = cell_summary(g)
            if s['is_t'] < 0 and s['oos_t'] < 0:  # BOTH splits negative
                veto.append((direction, cat, s))
    return veto


def main():
    out_dir = 'reports/findings/segments/retune_per_tier'
    os.makedirs(out_dir, exist_ok=True)
    all_recommendations = []
    md_lines = ['# Per-tier retune recommendations', '',
                 'Generated 2026-05-10. Validated on full IS+OOS trade history.', '',
                 'For each NMP-based tier, the optimal |z| band (E1) and the',
                 'structurally-negative (direction, 1h_z_se_category) cells (E2 VETO).',
                 '', '---', '']

    for tier in NMP_TIERS:
        print(f'\n=== {tier} ===')
        df = load_tier_trades(tier)
        if df.empty:
            print('  no trades'); continue
        n_is = int((df['split']=='IS').sum())
        n_oos = int((df['split']=='OOS').sum())
        print(f'  trades: {len(df):,} (IS {n_is}, OOS {n_oos})')

        # Baseline
        baseline = cell_summary(df)
        bt = baseline['total']; bi = baseline['is_t']; bo = baseline['oos_t']; bp = baseline['pf_wr']
        print(f'  baseline: total=${bt}  IS={bi}  OOS={bo}  PF_WR={bp}')

        # Optimal z-band
        best = find_optimal_band(df)
        if best is None:
            print(f'  no optimal band found');
            bt = baseline['total']
            md_lines.append(f'## {tier}')
            md_lines.append(f'')
            md_lines.append(f'baseline: total=${bt}  No optimal z-band found (no band with both IS and OOS positive).')
            md_lines.append(f'')
            continue
        lo, hi, e1_summary = best
        e1t = e1_summary['total']; e1i = e1_summary['is_t']; e1o = e1_summary['oos_t']; e1p = e1_summary['pf_wr']
        print(f'  E1 BAND |z| in [{lo}, {hi}]: total=${e1t}  IS={e1i}  OOS={e1o}  PF_WR={e1p}')

        # Attach 1h z_se to the band-filtered subset
        band_df = df[(df.z_abs >= lo) & (df.z_abs <= hi)]
        band_df = attach_1h_z(band_df)
        band_df = band_df.dropna(subset=['z_1h'])

        # Find veto cells WITHIN the band
        veto_cells = find_veto_cells(band_df)

        # Apply veto
        if veto_cells:
            veto_filter = band_df.copy()
            for direction, cat, _ in veto_cells:
                if direction == 'short':
                    mask = (veto_filter.direction == 'short')
                    if cat == 'aligned':
                        keep_cat = (veto_filter.z_1h < +0.3)
                    elif cat == 'opposed':
                        keep_cat = (veto_filter.z_1h > -0.3)
                    else:  # neutral
                        keep_cat = (veto_filter.z_1h.abs() >= 0.3)
                    veto_filter = veto_filter[~(mask & ~keep_cat)]
                else:
                    mask = (veto_filter.direction == 'long')
                    if cat == 'aligned':
                        keep_cat = (veto_filter.z_1h > -0.3)
                    elif cat == 'opposed':
                        keep_cat = (veto_filter.z_1h < +0.3)
                    else:
                        keep_cat = (veto_filter.z_1h.abs() >= 0.3)
                    veto_filter = veto_filter[~(mask & ~keep_cat)]
            combined = cell_summary(veto_filter)
            ct = combined['total']; ci = combined['is_t']; co = combined['oos_t']; cp = combined['pf_wr']
            print(f'  E1+E2 with vetoes:    total=${ct}  IS={ci}  OOS={co}  PF_WR={cp}')
        else:
            combined = e1_summary
            print(f'  no VETO cells found')

        rec = {
            'tier': tier,
            'baseline_total': baseline['total'],
            'baseline_is':    baseline['is_t'],
            'baseline_oos':   baseline['oos_t'],
            'baseline_pf':    baseline['pf_wr'],
            'z_floor':        lo,
            'z_ceiling':      hi,
            'e1_total':       e1_summary['total'],
            'e1_is':          e1_summary['is_t'],
            'e1_oos':         e1_summary['oos_t'],
            'e1_pf':          e1_summary['pf_wr'],
            'veto_cells':     ';'.join([f'{d}_{c}' for d,c,_ in veto_cells]),
            'combined_total': combined['total'],
            'combined_is':    combined['is_t'],
            'combined_oos':   combined['oos_t'],
            'combined_pf':    combined['pf_wr'],
            'uplift_oos':     round(combined['oos_t'] - baseline['oos_t'], 2),
        }
        all_recommendations.append(rec)

        bn = baseline['n']; bt = baseline['total']; bi = baseline['is_t']; bo = baseline['oos_t']; bp = baseline['pf_wr']
        en = e1_summary['n']; et = e1_summary['total']; ei = e1_summary['is_t']; eo = e1_summary['oos_t']; ep = e1_summary['pf_wr']
        cn = combined['n']; ct = combined['total']; ci = combined['is_t']; co = combined['oos_t']; cp = combined['pf_wr']
        ul = rec['uplift_oos']
        md_lines.append(f'## {tier}')
        md_lines.append('')
        md_lines.append(f'- baseline: n={bn}, total=${bt}, IS=${bi}, OOS=${bo}, PF_WR={bp}')
        md_lines.append(f'- **E1 band**: |z| in [{lo}, {hi}] -> n={en}, total=${et}, IS=${ei}, OOS=${eo}, PF_WR={ep}')
        if veto_cells:
            vlist = [(d,c) for d,c,_ in veto_cells]
            md_lines.append(f'- **E2 VETO** cells (both IS and OOS negative): {vlist}')
            md_lines.append(f'- **E1+E2 combined**: n={cn}, total=${ct}, IS=${ci}, OOS=${co}, PF_WR={cp}')
        else:
            md_lines.append(f'- E2 VETO: no cells qualify (no structural-loser cells found within band)')
        md_lines.append(f'- OOS uplift vs baseline: ${ul}')
        md_lines.append('')

    pd.DataFrame(all_recommendations).to_csv(
        os.path.join(out_dir, 'per_tier_retune.csv'), index=False)
    with open(os.path.join(out_dir, 'per_tier_retune.md'), 'w') as f:
        f.write('\n'.join(md_lines))
    print(f'\nSaved: {out_dir}/per_tier_retune.csv')
    print(f'Saved: {out_dir}/per_tier_retune.md')


if __name__ == '__main__':
    main()

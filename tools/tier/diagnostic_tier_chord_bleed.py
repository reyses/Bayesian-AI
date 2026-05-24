"""Diagnostic: where do existing iso tiers BLEED in chord/location space?

Original purpose of the primitives substrate (per 2026-05-09 lock): identify
the cells where existing tier strategies fail, so those cells become CONTEXT
FILTERS that condition tier firing.

Pipeline:
    1. Load all 9 tier IS + OOS pickles from training_iso_v2/output/
    2. For each trade, attach the 4-level chord shape at entry_ts
    3. Per (tier, chord_cell) compute mean $/trade IS, OOS, sign-match
    4. Identify BLEED CELLS — cells where:
         IS_mean < 0 AND OOS_mean < 0  (both losing, sign-stable)
         AND both n >= 10
    5. For each tier, output its top bleed cells — these are the
       context-filter candidates: "skip this tier when chord_cell ∈ bleed_set"

The bleed cells become input to a SkipFilter that wraps each tier in iso_v2.
A wrapped tier compose multiplicatively with existing context filters.

USAGE
    python tools/diagnostic_tier_chord_bleed.py
    python tools/diagnostic_tier_chord_bleed.py --min-n 10 --tiers FADE_CALM,RIDE_AGAINST
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TIERS = ['FADE_CALM', 'FADE_MOMENTUM', 'FADE_AGAINST',
          'RIDE_CALM', 'RIDE_MOMENTUM', 'RIDE_AGAINST',
          'KILL_SHOT', 'CASCADE', 'FREIGHT_TRAIN',
          'FADE_AT_BAND', 'NMP_FADE_RAW', 'NMP_RIDE_RAW']


def load_tier_trades(pickle_dir: str, tier: str, split: str) -> pd.DataFrame:
    path = os.path.join(pickle_dir, f'{split.lower()}_{tier}.pkl')
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, 'rb') as f:
        trades = pickle.load(f)
    rows = []
    for t in trades:
        rows.append({
            'tier':         tier,
            'split':        split,
            'direction':    t.direction,
            'entry_ts':     int(t.entry_ts),
            'exit_ts':      int(t.exit_ts),
            'entry_price':  t.entry_price,
            'exit_price':   t.exit_price,
            'pnl':          t.pnl,
            'peak_pnl':     t.peak_pnl,
            'entry_day':    t.entry_day,
            'exit_reason':  t.exit_reason,
            'entry_regime_idx': t.entry_regime_idx,
        })
    return pd.DataFrame(rows)


def attach_chord_shape(trades: pd.DataFrame, level_csv: str,
                        col_name: str) -> pd.DataFrame:
    seg = pd.read_csv(level_csv)
    seg['start_ts'] = seg['start_ts'].astype(np.int64)
    seg['end_ts']   = seg['end_ts'].astype(np.int64)
    out = trades.copy()
    out[col_name] = None
    for day, evs in tqdm(out.groupby('entry_day'),
                          desc=f'attach {col_name}'):
        day_segs = seg[seg['day'] == day]
        if day_segs.empty:
            continue
        day_segs = day_segs.sort_values('start_ts').reset_index(drop=True)
        starts = day_segs['start_ts'].values
        ends   = day_segs['end_ts'].values
        shapes = day_segs['shape'].values
        ts = evs['entry_ts'].astype(np.int64).values
        cand = np.searchsorted(starts, ts, side='right') - 1
        cand = np.clip(cand, 0, len(starts) - 1)
        valid = ts <= ends[cand]
        result = np.where(valid, shapes[cand], None)
        out.loc[evs.index, col_name] = result
    return out


def per_cell_stats(df: pd.DataFrame, key_cols: list[str],
                    min_n: int = 10) -> pd.DataFrame:
    """For each (tier × *key_cols) cell, IS mean/n + OOS mean/n + sign-match."""
    rows = []
    for tier, tdf in df.groupby('tier'):
        for keys, g in tdf.groupby(key_cols):
            is_g  = g[g['split'] == 'IS']
            oos_g = g[g['split'] == 'OOS']
            n_is  = len(is_g);  n_oos = len(oos_g)
            if n_is < min_n and n_oos < min_n:
                continue
            rec = {'tier': tier}
            for c, v in zip(key_cols, keys if isinstance(keys, tuple) else (keys,)):
                rec[c] = v
            rec['n_is'] = n_is; rec['n_oos'] = n_oos
            rec['mean_is']  = round(float(is_g['pnl'].mean()), 2)  if n_is  > 0 else np.nan
            rec['mean_oos'] = round(float(oos_g['pnl'].mean()), 2) if n_oos > 0 else np.nan
            rec['sum_is']   = round(float(is_g['pnl'].sum()), 0)   if n_is  > 0 else 0
            rec['sum_oos']  = round(float(oos_g['pnl'].sum()), 0)  if n_oos > 0 else 0
            mi = rec.get('mean_is', np.nan)
            mo = rec.get('mean_oos', np.nan)
            if pd.notna(mi) and pd.notna(mo):
                rec['sign_match'] = int(np.sign(mi) == np.sign(mo))
            else:
                rec['sign_match'] = -1
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle-dir', default='training_iso_v2/output')
    ap.add_argument('--seg-dir', default='reports/findings/segments/simple_bulk_v2')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/diagnostic_tier_bleed')
    ap.add_argument('--min-n', type=int, default=10)
    ap.add_argument('--tiers', default=None,
                     help='Comma-separated subset; default all available')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tiers = args.tiers.split(',') if args.tiers else TIERS
    print(f'Loading tier trades from {args.pickle_dir} for {len(tiers)} tiers...')
    all_trades = []
    for tier in tiers:
        for split in ['IS', 'OOS']:
            df = load_tier_trades(args.pickle_dir, tier, split)
            if not df.empty:
                all_trades.append(df)
                print(f'  {split:<3s} {tier:<16s} n={len(df):,}  '
                       f'mean_pnl={df["pnl"].mean():+.2f}  '
                       f'sum={df["pnl"].sum():+.0f}')
    if not all_trades:
        print('No trade pickles found.'); return
    trades = pd.concat(all_trades).reset_index(drop=True)
    print(f'\nTotal trades: {len(trades):,}  '
           f'(IS: {(trades["split"]=="IS").sum():,}, '
           f'OOS: {(trades["split"]=="OOS").sum():,})')

    # Attach 4-level chord at entry_ts
    for level, col in [('phrases', 'phrase_shape'),
                        ('motifs', 'motif_shape'),
                        ('sub_motifs', 'sub_motif_shape'),
                        ('measures', 'measure_shape')]:
        path = os.path.join(args.seg_dir, f'all_{level}.csv')
        trades = attach_chord_shape(trades, path, col)

    # Coverage
    chord_cols = ['phrase_shape', 'motif_shape', 'sub_motif_shape', 'measure_shape']
    n_full = trades[chord_cols].notna().all(axis=1).sum()
    print(f'\nTrades with full 4-level chord: {n_full:,} of {len(trades):,} '
           f'({100*n_full/len(trades):.1f}%)')

    # Save merged substrate
    sub_path = os.path.join(args.out_dir, 'all_trades_with_chord.parquet')
    trades.to_parquet(sub_path, index=False)
    print(f'Per-trade substrate -> {sub_path}')

    # ─── Per-tier baselines ───
    print(f'\n=== PER-TIER OVERALL (IS, OOS) ===')
    base = (trades.groupby(['tier', 'split'])
                  .agg(n=('pnl', 'size'),
                        mean=('pnl', 'mean'),
                        sum=('pnl', 'sum'))
                  .round(2).reset_index())
    print(base.to_string(index=False))
    base.to_csv(os.path.join(args.out_dir, 'tier_baselines.csv'), index=False)

    # ─── Per (tier × measure_shape) cell stats ───
    print(f'\n=== TIER x MEASURE_SHAPE cells (min_n={args.min_n}) ===')
    cell_m = per_cell_stats(trades, ['measure_shape'], args.min_n)
    cell_m.to_csv(os.path.join(args.out_dir, 'cells_tier_x_measure.csv'),
                   index=False)
    # Bleed cells: both IS and OOS negative AND sign-match
    bleed_m = cell_m[(cell_m['mean_is'] < 0) & (cell_m['mean_oos'] < 0)
                     & (cell_m['sign_match'] == 1)
                     & (cell_m['n_is'] >= args.min_n)
                     & (cell_m['n_oos'] >= args.min_n)].copy()
    bleed_m['total_bleed'] = bleed_m['sum_is'] + bleed_m['sum_oos']
    bleed_m = bleed_m.sort_values('total_bleed')
    print(f'Bleed cells (IS<0 AND OOS<0, sign-match): {len(bleed_m)}')
    if not bleed_m.empty:
        print(bleed_m.head(20).to_string(index=False))
    bleed_m.to_csv(os.path.join(args.out_dir, 'BLEED_tier_x_measure.csv'),
                    index=False)

    # ─── Per (tier × sub_motif × measure) cell stats ───
    print(f'\n=== TIER x SUB_MOTIF x MEASURE cells (min_n={args.min_n}) ===')
    cell_sm = per_cell_stats(trades, ['sub_motif_shape', 'measure_shape'], args.min_n)
    cell_sm.to_csv(os.path.join(args.out_dir, 'cells_tier_x_sub_motif_x_measure.csv'),
                    index=False)
    bleed_sm = cell_sm[(cell_sm['mean_is'] < 0) & (cell_sm['mean_oos'] < 0)
                        & (cell_sm['sign_match'] == 1)
                        & (cell_sm['n_is'] >= args.min_n)
                        & (cell_sm['n_oos'] >= args.min_n)].copy()
    bleed_sm['total_bleed'] = bleed_sm['sum_is'] + bleed_sm['sum_oos']
    bleed_sm = bleed_sm.sort_values('total_bleed')
    print(f'Bleed cells: {len(bleed_sm)}')
    if not bleed_sm.empty:
        print(bleed_sm.head(20).to_string(index=False))
    bleed_sm.to_csv(os.path.join(args.out_dir, 'BLEED_tier_x_sub_motif_x_measure.csv'),
                     index=False)

    # ─── PER-TIER bleed contribution: how much $ would skipping these cells save? ───
    print(f'\n=== PER-TIER BLEED CONTRIBUTION (would save by skipping bleed cells) ===')
    summary_rows = []
    for tier in trades['tier'].unique():
        t_total_is = trades[(trades['tier']==tier) & (trades['split']=='IS')]['pnl'].sum()
        t_total_oos = trades[(trades['tier']==tier) & (trades['split']=='OOS')]['pnl'].sum()
        # bleed at measure_shape level
        b_m = bleed_m[bleed_m['tier']==tier]
        bleed_m_is = b_m['sum_is'].sum() if not b_m.empty else 0
        bleed_m_oos = b_m['sum_oos'].sum() if not b_m.empty else 0
        # at sub_motif x measure level
        b_sm = bleed_sm[bleed_sm['tier']==tier]
        bleed_sm_is = b_sm['sum_is'].sum() if not b_sm.empty else 0
        bleed_sm_oos = b_sm['sum_oos'].sum() if not b_sm.empty else 0
        summary_rows.append({
            'tier': tier,
            'IS_total': round(float(t_total_is), 0),
            'OOS_total': round(float(t_total_oos), 0),
            'bleed_meas_n': len(b_m),
            'bleed_meas_IS': round(float(bleed_m_is), 0),
            'bleed_meas_OOS': round(float(bleed_m_oos), 0),
            'bleed_subm_n': len(b_sm),
            'bleed_subm_IS': round(float(bleed_sm_is), 0),
            'bleed_subm_OOS': round(float(bleed_sm_oos), 0),
            'IS_after_skip_subm': round(float(t_total_is - bleed_sm_is), 0),
            'OOS_after_skip_subm': round(float(t_total_oos - bleed_sm_oos), 0),
        })
    sdf = pd.DataFrame(summary_rows).sort_values('OOS_after_skip_subm',
                                                    ascending=False)
    print(sdf.to_string(index=False))
    sdf.to_csv(os.path.join(args.out_dir, 'per_tier_skip_savings.csv'),
                index=False)

    print(f'\nAll artifacts -> {args.out_dir}')


if __name__ == '__main__':
    main()

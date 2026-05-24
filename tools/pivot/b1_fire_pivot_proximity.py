"""B1 fire-to-pivot proximity diagnostic.

User question (2026-05-17): the B3 regressor failed because it had to predict
for every bar, but B1 K=1 binary at thr=0.70 hits 13.6% precision (2.61x lift)
with sparse fires. Are those sparse fires CONCENTRATED near real pivots?

This script answers two questions:

  Q1. PROXIMITY: for each B1 high-confidence fire, what's the distance to
      the nearest actual pivot (forward)? Stats: median, p25, p75, % within
      K minutes for K=1,3,5,10.

  Q2. EVENT STUDY: stack all real pivots, compute the AVERAGE B1 probability
      profile from t=-30min before pivot through t=+5min after. If P rises
      monotonically and peaks ~30s before pivot, the bridge mechanism is
      working as theorized.

Compares to a RANDOM-POINT baseline to confirm the signal is real and not
just artifact of probability distribution.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def pivot_centroid_ts_per_day(truth_df: pd.DataFrame) -> dict:
    """Same as B1: collapse is_pivot==1 runs to centroid timestamps."""
    out = {}
    for day, g in truth_df.groupby('day'):
        piv = g[g['is_pivot'] == 1].sort_values('timestamp')
        if len(piv) == 0:
            out[day] = np.array([], dtype=np.int64); continue
        ts = piv['timestamp'].values.astype(np.int64)
        groups = [[ts[0]]]
        for i in range(1, len(ts)):
            if ts[i] - ts[i-1] > 90:
                groups.append([ts[i]])
            else:
                groups[-1].append(ts[i])
        out[day] = np.array([int(np.median(g)) for g in groups], dtype=np.int64)
    return out


def proximity_to_next_pivot(bar_ts: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    """For each bar ts, return seconds to the NEXT pivot at or after it
    (NaN if no future pivot)."""
    out = np.full(len(bar_ts), np.nan)
    if len(pivots) == 0:
        return out
    idx = np.searchsorted(pivots, bar_ts, side='left')
    valid = idx < len(pivots)
    out[valid] = (pivots[idx[valid]] - bar_ts[valid]).astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b1-cache',
                    default='reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet')
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b1_fire_pivot_proximity.txt')
    ap.add_argument('--out-csv',
                    default='reports/findings/regret_oracle/b1_fire_pivot_proximity.csv')
    args = ap.parse_args()

    print('Loading B1 + truth...')
    b1 = pd.read_parquet(args.b1_cache).sort_values(['day', 'timestamp']).reset_index(drop=True)
    tr = pd.read_parquet(args.truth).sort_values(['day', 'timestamp']).reset_index(drop=True)
    print(f'  B1: {len(b1):,}   truth: {len(tr):,}')

    # Compute per-bar forward distance to next pivot
    print('Computing per-bar forward distance to next pivot...')
    pivots_per_day = pivot_centroid_ts_per_day(tr)
    b1['fwd_pivot_s'] = np.nan
    for day, g in b1.groupby('day'):
        pivs = pivots_per_day.get(day, np.array([], dtype=np.int64))
        bar_ts = g['timestamp'].values.astype(np.int64)
        b1.loc[g.index, 'fwd_pivot_s'] = proximity_to_next_pivot(bar_ts, pivs)
    b1['fwd_pivot_min'] = b1['fwd_pivot_s'] / 60.0

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B1 FIRE-TO-PIVOT PROXIMITY  (NT8 OOS, 32 days)')
    out('=' * 78)
    out(f'Total bars: {len(b1):,}   bars with future pivot: {b1["fwd_pivot_s"].notna().sum():,}')
    out('')

    # ============================================================
    # Q1: PROXIMITY -- where do B1 high-confidence fires land?
    # ============================================================
    out('Q1. PROXIMITY: where do B1 high-confidence fires land?')
    out('-' * 78)
    out(f'  {"K":>4}  {"thr":>5}  {"n_fires":>8}  '
        f'{"fwd_med_min":>12}  {"%w/piv<1m":>10}  {"%<3m":>6}  '
        f'{"%<5m":>6}  {"%<10m":>7}')

    rows = []
    for K in [1, 3, 5, 10]:
        col = f'p_pivot_{K}m'
        for thr in [0.50, 0.70, 0.85, 0.95]:
            mask = (b1[col] >= thr) & b1['fwd_pivot_s'].notna()
            n = int(mask.sum())
            if n == 0:
                out(f'  {K:>4}  {thr:>5.2f}  {n:>8,}  -')
                continue
            ttn_min = b1.loc[mask, 'fwd_pivot_min'].values
            med = float(np.median(ttn_min))
            pct_1m = float((ttn_min < 1).mean() * 100)
            pct_3m = float((ttn_min < 3).mean() * 100)
            pct_5m = float((ttn_min < 5).mean() * 100)
            pct_10m = float((ttn_min < 10).mean() * 100)
            out(f'  {K:>4}  {thr:>5.2f}  {n:>8,}  '
                f'{med:>11.2f}m  {pct_1m:>9.1f}%  {pct_3m:>5.1f}%  '
                f'{pct_5m:>5.1f}%  {pct_10m:>6.1f}%')
            rows.append({'K': K, 'thr': thr, 'n_fires': n,
                          'fwd_med_min': med, 'pct_1m': pct_1m,
                          'pct_3m': pct_3m, 'pct_5m': pct_5m,
                          'pct_10m': pct_10m})

    out('')
    out('Baseline (all bars, regardless of B1 prob):')
    base = b1[b1['fwd_pivot_s'].notna()]
    base_ttn = base['fwd_pivot_min'].values
    out(f'  median={np.median(base_ttn):.2f}m   '
        f'%<1m={(base_ttn < 1).mean()*100:.1f}%   '
        f'%<3m={(base_ttn < 3).mean()*100:.1f}%   '
        f'%<5m={(base_ttn < 5).mean()*100:.1f}%   '
        f'%<10m={(base_ttn < 10).mean()*100:.1f}%')
    out('')
    out('Interpretation:')
    out('  - LIFT = (B1-fire pct < 10m) / (baseline pct < 10m).')
    out('  - If LIFT >> 1 at high thr, fires concentrate near pivots = signal is real.')
    out('  - If LIFT ~ 1, fires scattered randomly = no usable signal.')

    # ============================================================
    # Q2: EVENT STUDY -- B1 probability profile around real pivots
    # ============================================================
    out('')
    out('Q2. EVENT STUDY: average B1 probability vs seconds-from-pivot')
    out('-' * 78)
    out('  Stacks all real pivots, computes mean(P) per relative-time bin.')
    out('  If P rises monotonically as we approach pivot at t=0, the')
    out('  bridge mechanism is mathematically present in the model.')
    out('')

    # Build event window per real pivot
    bins_s = np.arange(-1800, 301, 60)   # -30min to +5min in 60s bins
    bin_centers = (bins_s[:-1] + bins_s[1:]) / 2
    sums = {K: np.zeros(len(bin_centers)) for K in [1, 3, 5, 10]}
    counts = np.zeros(len(bin_centers))

    print('Computing event-study profile per pivot...')
    for day, g in tqdm(b1.groupby('day'), desc='days'):
        pivs = pivots_per_day.get(day, np.array([], dtype=np.int64))
        if len(pivs) == 0:
            continue
        ts_arr = g['timestamp'].values.astype(np.int64)
        for p in pivs:
            rel = ts_arr - p
            in_win = (rel >= bins_s[0]) & (rel < bins_s[-1])
            if not in_win.any():
                continue
            bin_idx = np.digitize(rel[in_win], bins_s) - 1
            for K in [1, 3, 5, 10]:
                vals = g.loc[g.index[in_win], f'p_pivot_{K}m'].values
                np.add.at(sums[K], bin_idx, vals)
            np.add.at(counts, bin_idx, 1)

    out(f'  {"window":>14}  {"n_pivots":>8}  '
        f'{"P_K=1":>7}  {"P_K=3":>7}  {"P_K=5":>7}  {"P_K=10":>7}')
    for i, (lo, hi) in enumerate(zip(bins_s[:-1], bins_s[1:])):
        if counts[i] == 0:
            continue
        lo_str = f'[{lo/60:+.0f}m,{hi/60:+.0f}m)'
        means = {K: sums[K][i] / counts[i] for K in [1, 3, 5, 10]}
        out(f'  {lo_str:>14}  {int(counts[i]):>8,}  '
            f'{means[1]:>6.3f}  {means[3]:>6.3f}  '
            f'{means[5]:>6.3f}  {means[10]:>6.3f}')

    out('')
    out('Read the columns top-to-bottom:')
    out('  - If P_K=1 starts low at -30m and ramps up to peak near 0m,')
    out('    the model is correctly detecting the rising pivot signal.')
    out('  - The slope steepness tells us HOW EARLY the model sees it.')
    out('  - P should DROP sharply after t=0 (post-pivot, no longer imminent).')

    # Save outputs
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    # Also save event-study data
    event_df = pd.DataFrame({
        'bin_lo_s': bins_s[:-1],
        'bin_hi_s': bins_s[1:],
        'bin_center_s': bin_centers,
        'n_pivots_in_bin': counts.astype(int),
    })
    for K in [1, 3, 5, 10]:
        event_df[f'mean_p_{K}m'] = np.where(counts > 0, sums[K] / np.maximum(counts, 1), np.nan)
    event_df.to_csv(Path(args.out_csv).with_suffix('.event_study.csv'), index=False)

    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_report}')
    print(f'Wrote: {args.out_csv}')
    print(f'Wrote: {Path(args.out_csv).with_suffix(".event_study.csv")}')


if __name__ == '__main__':
    main()

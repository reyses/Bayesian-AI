"""Exploratory: tag oracle entries that sit inside an OSCILLATION CLUSTER.

Per user 2026-05-14: "catch the trades that have low PNL but are oscillating
kinda like the picture" — i.e., the dense direction-flipping chop bands we
saw on 2025-07-08. The goal here is to *identify* those clusters so we can
later experiment with fusing them. Exploratory — outputs a per-entry CSV
with oscillation-window features + a cluster-candidate flag.

For each entry i, look at the window [t_i - W, t_i + W] within the same
session, and compute:
    osc_window_n         number of oracle entries in the window
    osc_price_range_pts  spread of entry prices in points
    osc_price_range_$    same in dollars (× $2/point MNQ)
    osc_n_long           count of LONG entries in window
    osc_n_short          count of SHORT entries
    osc_flip_rate        (#direction-changes) / (n-1) — 1.0 = alternating
    osc_mean_mfe_$       mean MFE of entries in window
    osc_sum_long_mfe_$   sum of LONG-side MFE (one-side excursion bound)
    osc_sum_short_mfe_$  sum of SHORT-side MFE
    osc_window_span_min  actual time span from first→last entry in window

Candidate flag (`is_oscillation_candidate`):
    window_n >= MIN_N
    AND price_range_pts <= MAX_SPREAD_PTS
    AND mfe_dollars < PNL_THRESHOLD
    AND both LONG and SHORT present in window  (= mixed-direction chop)

Not a final fusion rule — exploratory feature engineering so we can look at
which entries cluster, where, and what fusion COULD recover.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path('reports/findings/regret_oracle')
DOLLAR_PER_POINT = 2.0   # MNQ


def compute_oscillation_features(df: pd.DataFrame, window_min: float) -> pd.DataFrame:
    df = df.sort_values('oracle_ts').reset_index(drop=True)
    ts        = df['oracle_ts'].values.astype(np.int64)
    prices    = df['entry_price'].values.astype(float)
    mfes      = df['mfe_dollars'].values.astype(float)
    dirs      = df['direction'].values
    if 'session_id' in df.columns:
        sids = df['session_id'].values.astype(np.int64)
    else:
        sids = np.zeros(len(df), dtype=np.int64)

    window_s = int(window_min * 60)
    feats = []
    for i in range(len(df)):
        lo, hi = ts[i] - window_s, ts[i] + window_s
        mask = (ts >= lo) & (ts <= hi) & (sids == sids[i])
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            feats.append({})
            continue
        sub_prices = prices[idxs]
        sub_dirs   = dirs[idxs]
        sub_mfes   = mfes[idxs]

        order = np.argsort(ts[idxs])
        sub_dirs_t = sub_dirs[order]
        flips = int((sub_dirs_t[1:] != sub_dirs_t[:-1]).sum()) if len(sub_dirs_t) > 1 else 0

        price_range_pts = float(sub_prices.max() - sub_prices.min())
        long_mask  = (sub_dirs == 'LONG')
        short_mask = (sub_dirs == 'SHORT')
        feats.append({
            'osc_window_n':         len(idxs),
            'osc_price_range_pts':  round(price_range_pts, 2),
            'osc_price_range_$':    round(price_range_pts * DOLLAR_PER_POINT, 2),
            'osc_n_long':           int(long_mask.sum()),
            'osc_n_short':          int(short_mask.sum()),
            'osc_flip_rate':        round(flips / max(len(idxs) - 1, 1), 3),
            'osc_mean_mfe_$':       round(float(sub_mfes.mean()), 2),
            'osc_sum_long_mfe_$':   round(float(sub_mfes[long_mask].sum()), 2),
            'osc_sum_short_mfe_$':  round(float(sub_mfes[short_mask].sum()), 2),
            'osc_window_span_min':  round(float(ts[idxs].max() - ts[idxs].min()) / 60.0, 1),
        })
    return pd.concat([df, pd.DataFrame(feats)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--window-min', type=float, default=30,
                    help='±N minutes window for oscillation detection (default 30)')
    ap.add_argument('--pnl-threshold', type=float, default=30,
                    help='Entries with mfe < this are candidates (default $30)')
    ap.add_argument('--max-spread-pts', type=float, default=10,
                    help='Max price spread across window entries, in points '
                         '(default 10 pts = $20)')
    ap.add_argument('--min-n', type=int, default=3,
                    help='Minimum entries in window to call it an oscillation (default 3)')
    ap.add_argument('--name', default='IS_full')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} entries from {args.input}')
    print(f'Params: window=±{args.window_min}min  pnl<${args.pnl_threshold}  '
          f'max_spread={args.max_spread_pts}pts  min_n={args.min_n}')

    out = compute_oscillation_features(df, window_min=args.window_min)

    # Candidate flag: mixed-direction chop cluster with low PNL
    out['is_oscillation_candidate'] = (
        (out['osc_window_n']        >= args.min_n) &
        (out['osc_price_range_pts'] <= args.max_spread_pts) &
        (out['mfe_dollars']         <  args.pnl_threshold) &
        (out['osc_n_long']          >= 1) &
        (out['osc_n_short']         >= 1)
    )

    out_path = out_dir / f'oscillation_explorer_{args.name}.csv'
    out.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}')

    n_total = len(out)
    n_cand  = int(out['is_oscillation_candidate'].sum())
    n_lowpnl = int((out['mfe_dollars'] < args.pnl_threshold).sum())
    print(f'\n=== Summary ===')
    print(f'  Total entries           : {n_total}')
    print(f'  Low-PNL entries (<${args.pnl_threshold}): {n_lowpnl}  ({100*n_lowpnl/n_total:.1f}%)')
    print(f'  Oscillation candidates  : {n_cand}  ({100*n_cand/n_total:.1f}% of all,  '
          f'{100*n_cand/max(n_lowpnl,1):.1f}% of low-PNL)')

    if n_cand == 0:
        print('\n  No candidates at these thresholds — try wider --window-min '
              'or larger --max-spread-pts.')
        return

    cands = out[out['is_oscillation_candidate']]
    print(f'\n=== Candidate distribution ===')
    print(f'  osc_window_n     : median {int(cands.osc_window_n.median())}  '
          f'max {int(cands.osc_window_n.max())}')
    print(f'  price_range_pts  : median {cands.osc_price_range_pts.median():.1f}  '
          f'max {cands.osc_price_range_pts.max():.1f}')
    print(f'  flip_rate        : median {cands.osc_flip_rate.median():.2f}  '
          f'(1.0 = perfectly alternating)')
    print(f'  mean_mfe in window: median ${cands["osc_mean_mfe_$"].median():.0f}')
    print(f'  direction split  : {(cands.direction == "LONG").sum()} LONG / '
          f'{(cands.direction == "SHORT").sum()} SHORT')

    if 'tod_minutes' in cands.columns:
        tb = (cands.tod_minutes // 240).astype(int)
        print(f'\n  By tod-bucket (4h blocks since session open):')
        for k, v in tb.value_counts().sort_index().items():
            label = f'{k*4}-{(k+1)*4}h'
            print(f'    {label:>8s}: {v}')

    if 'session_id' in cands.columns:
        sess_counts = cands.groupby('session_id').size().sort_values(ascending=False)
        print(f'\n  Top 10 sessions by candidate count (which days are oscillation-heavy?):')
        for sid, n in sess_counts.head(10).items():
            sess_date = (cands[cands.session_id == sid]['session_date'].iloc[0]
                         if 'session_date' in cands.columns else '?')
            print(f'    session {sid} ({sess_date}): {n} candidates')

    # Fusion sanity: in candidate windows, is there a one-sided opportunity hiding?
    # If sum_long >> sum_short (or vice versa) in a cluster, fusion might recover
    # a single directional move. If both sides are similar, it's true chop.
    print(f'\n=== Fusion preview (per-entry one-sidedness) ===')
    cands_c = cands.copy()
    side_imbalance = (cands_c['osc_sum_long_mfe_$']
                      - cands_c['osc_sum_short_mfe_$']).abs()
    side_total = (cands_c['osc_sum_long_mfe_$']
                  + cands_c['osc_sum_short_mfe_$'])
    cands_c['imbalance_ratio'] = (side_imbalance / side_total.replace(0, np.nan))
    print(f'  imbalance_ratio = |sum_long - sum_short| / (sum_long + sum_short)')
    print(f'  median: {cands_c["imbalance_ratio"].median():.2f}  '
          f'p25 {cands_c["imbalance_ratio"].quantile(.25):.2f}  '
          f'p75 {cands_c["imbalance_ratio"].quantile(.75):.2f}')
    print(f'  (low = both sides balanced = true chop;  '
          f'high = one side dominates = directional move broken up)')
    one_sided = cands_c[cands_c['imbalance_ratio'] > 0.6]
    print(f'  Strongly one-sided (>0.6 imbalance): {len(one_sided)}  '
          f'({100*len(one_sided)/len(cands_c):.0f}% of candidates)')


if __name__ == '__main__':
    main()

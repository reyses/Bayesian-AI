"""MFE mode per (tier × regime) cell from iso pickles.

For each ClosedTrade in training_iso_v2/output/is_<TIER>.pkl, reads
peak_pnl (the trade's MFE — maximum favorable excursion in $) and
entry_regime_idx, then computes per-cell:
    n, MFE_mode (bin=$5), MFE_p25/p50/p75/p90, MFE_mean,
    realized_pnl_mode, realized_mean, capture_ratio = mean_realized / mean_mfe.

Usage:
    python tools/iso_mfe_by_regime.py
    python tools/iso_mfe_by_regime.py --prefix training_iso_v2/output/oos
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso_v2.state import REGIME_VOCAB


def histmode(v, bw):
    v = np.asarray([x for x in v if x == x], dtype=np.float64)
    if len(v) == 0:
        return float('nan')
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < bw:
        return float(np.median(v))
    n = max(1, int(np.ceil((hi - lo) / bw)))
    edges = np.linspace(lo, hi, n + 1)
    counts, _ = np.histogram(v, bins=edges)
    j = int(np.argmax(counts))
    return float((edges[j] + edges[j + 1]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='training_iso_v2/output/is')
    ap.add_argument('--bin', type=float, default=5.0,
                          help='Histogram bin width $ for MFE mode')
    ap.add_argument('--out-csv',
                          default='reports/findings/iso_mfe_by_regime.csv')
    args = ap.parse_args()

    pkls = sorted(glob.glob(f'{args.prefix}_*.pkl'))
    pkls = [p for p in pkls if not p.endswith('_regret.pkl')
                  and 'summary' not in p]
    if not pkls:
        print(f'No tier pickles found at {args.prefix}_*.pkl')
        return

    rows = []
    for path in pkls:
        tier = os.path.basename(path).replace('.pkl', '').split('_', 1)[1]
        with open(path, 'rb') as f:
            trades = pickle.load(f)
        if not trades:
            continue

        # Group by regime_idx -> map to label
        by_reg = defaultdict(list)
        for t in trades:
            by_reg[t.entry_regime_idx].append(t)

        # All-regime row
        mfe_all = np.array([t.peak_pnl for t in trades])
        pnl_all = np.array([t.pnl for t in trades])
        rows.append({
            'tier': tier,
            'regime': 'ALL',
            'n': len(trades),
            'mfe_mode': histmode(mfe_all, args.bin),
            'mfe_mean': float(mfe_all.mean()),
            'mfe_p25': float(np.percentile(mfe_all, 25)),
            'mfe_p50': float(np.percentile(mfe_all, 50)),
            'mfe_p75': float(np.percentile(mfe_all, 75)),
            'mfe_p90': float(np.percentile(mfe_all, 90)),
            'pnl_mode': histmode(pnl_all, 2.0),
            'pnl_mean': float(pnl_all.mean()),
            'capture': float(pnl_all.mean() / mfe_all.mean())
                            if mfe_all.mean() > 0 else float('nan'),
        })
        for ridx, ts in sorted(by_reg.items()):
            mfe = np.array([t.peak_pnl for t in ts])
            pnl = np.array([t.pnl for t in ts])
            label = REGIME_VOCAB[ridx] if 0 <= ridx < len(REGIME_VOCAB) else f'IDX{ridx}'
            rows.append({
                'tier': tier,
                'regime': label,
                'n': len(ts),
                'mfe_mode': histmode(mfe, args.bin),
                'mfe_mean': float(mfe.mean()),
                'mfe_p25': float(np.percentile(mfe, 25)),
                'mfe_p50': float(np.percentile(mfe, 50)),
                'mfe_p75': float(np.percentile(mfe, 75)),
                'mfe_p90': float(np.percentile(mfe, 90)),
                'pnl_mode': histmode(pnl, 2.0),
                'pnl_mean': float(pnl.mean()),
                'capture': float(pnl.mean() / mfe.mean())
                                if mfe.mean() > 0 else float('nan'),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Pretty print per-tier × regime
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.float_format', lambda x: f'{x:>+8.2f}')

    tiers = sorted(df['tier'].unique())
    print(f'\n{"=" * 138}')
    print(f'MFE BY (TIER × REGIME)   bin=${args.bin}   prefix={args.prefix}')
    print(f'  MFE = peak_pnl per trade (max favorable excursion in $)')
    print(f'  capture = mean_realized / mean_mfe   (closer to 1 = exits captured the move)')
    print(f'{"=" * 138}')
    fmt = ('{tier:<14} {regime:<13} {n:>6}  '
              '{mfe_mode:>+8.2f} {mfe_mean:>+8.2f}  '
              '{mfe_p25:>+8.2f} {mfe_p50:>+8.2f} {mfe_p75:>+8.2f} {mfe_p90:>+8.2f} | '
              '{pnl_mode:>+8.2f} {pnl_mean:>+8.2f}  {capture:>+6.2f}')
    print('{:<14} {:<13} {:>6}  {:>8} {:>8}  {:>8} {:>8} {:>8} {:>8} | '
              '{:>8} {:>8}  {:>6}'.format(
        'tier', 'regime', 'n',
        'mfeMode', 'mfeMean', 'mfeQ25', 'mfeQ50', 'mfeQ75', 'mfeQ90',
        'pnlMode', 'pnlMean', 'capt'))
    print('-' * 138)
    for tier in tiers:
        sub = df[df['tier'] == tier]
        # Put ALL row first, then individual regimes by n descending
        all_row = sub[sub['regime'] == 'ALL']
        rest = sub[sub['regime'] != 'ALL'].sort_values('n', ascending=False)
        for _, r in pd.concat([all_row, rest]).iterrows():
            print(fmt.format(**r.to_dict()))
        print('-' * 138)
    print(f'\nCSV saved -> {args.out_csv}')


if __name__ == '__main__':
    main()

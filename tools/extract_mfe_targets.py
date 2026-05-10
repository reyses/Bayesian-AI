"""Extract per-cell modal MFE PnL ($) — the price-target derived from data.

Reads the per-trade trajectory parquet (or the simpler peak-signature parquet)
and computes per (tier × regime × direction) the modal MFE PnL in dollars.
That's the empirical "where do peaks happen" target, used by the
MFEPriceTarget exit rule.

Output:
    training_iso_v2/output/mfe_targets_per_cell.json
        {cell_key: {mfe_mode, mfe_q25, mfe_q50, mfe_q75, mfe_q90, mfe_mean, n}}

Usage:
    python tools/extract_mfe_targets.py
    python tools/extract_mfe_targets.py --parquet <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso_v2.state import REGIME_VOCAB


def histmode(v: np.ndarray, bin_width: float = 5.0) -> float:
    """Histogram-mode for $ values, bin width in $ (default $5)."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < bin_width:
        return float(np.median(v))
    n = max(1, int(np.ceil((hi - lo) / bin_width)))
    edges = np.linspace(lo, hi, n + 1)
    counts, _ = np.histogram(v, bins=edges)
    j = int(np.argmax(counts))
    return float((edges[j] + edges[j + 1]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet',
                          default='reports/findings/peak_signatures/trade_mfe_features.parquet')
    ap.add_argument('--out',
                          default='training_iso_v2/output/mfe_targets_per_cell.json')
    ap.add_argument('--bin', type=float, default=5.0,
                          help='Histogram bin in $ (default $5)')
    ap.add_argument('--min-n', type=int, default=30,
                          help='Min trades per cell to compute target (default 30)')
    ap.add_argument('--target-quantile', type=float, default=0.50,
                          help='Quantile to use as target (default 0.50 = median '
                                 'peak; 0.40 = more conservative; 0.65 = greedier)')
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if 'mfe_pnl' not in df.columns:
        print(f'!!! parquet has no mfe_pnl column. Available: {list(df.columns)[:30]}')
        sys.exit(1)

    print(f'Loaded {len(df)} rows from {args.parquet}')

    out = {
        '_meta': {
            'method': 'per (tier × regime × direction) modal MFE PnL ($)',
            'bin_width_usd': args.bin,
            'min_n_per_cell': args.min_n,
            'target_quantile_used': args.target_quantile,
        },
    }
    summary_rows = []

    for (tier, ridx, direction), sub in df.groupby(['tier', 'regime_idx', 'direction']):
        if len(sub) < args.min_n:
            continue
        regime = (REGIME_VOCAB[ridx] if 0 <= ridx < len(REGIME_VOCAB)
                       else f'IDX{ridx}')
        cell_key = f'{tier}|{regime}|{direction}'

        mfe = sub['mfe_pnl'].values
        mfe = mfe[np.isfinite(mfe)]
        if len(mfe) < args.min_n:
            continue

        target = float(np.quantile(mfe, args.target_quantile))
        out[cell_key] = {
            'n': int(len(mfe)),
            'mfe_mode': histmode(mfe, args.bin),
            'mfe_mean': float(mfe.mean()),
            'mfe_q25': float(np.quantile(mfe, 0.25)),
            'mfe_q50': float(np.quantile(mfe, 0.50)),
            'mfe_q75': float(np.quantile(mfe, 0.75)),
            'mfe_q90': float(np.quantile(mfe, 0.90)),
            'price_target_usd': target,
        }
        summary_rows.append({
            'cell': cell_key, 'tier': tier, 'regime': regime,
            'direction': direction, 'n': int(len(mfe)),
            'mode': out[cell_key]['mfe_mode'],
            'q25': out[cell_key]['mfe_q25'],
            'q50': out[cell_key]['mfe_q50'],
            'q75': out[cell_key]['mfe_q75'],
            'q90': out[cell_key]['mfe_q90'],
            'mean': out[cell_key]['mfe_mean'],
            'target': target,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nMFE targets JSON -> {args.out}')

    # Pretty-print
    print(f'\n{"=" * 110}')
    print(f'PER-CELL MFE TARGETS  (price target in $, derived from peak distributions)')
    print(f'  target = quantile {args.target_quantile} of MFE_pnl distribution')
    print(f'{"=" * 110}')
    print(f'{"cell":<45} {"n":>5} {"mode":>7} {"q25":>7} {"q50":>7} '
              f'{"q75":>7} {"q90":>7} {"mean":>7} {"TARGET":>8}')
    print('-' * 110)
    summary_rows.sort(key=lambda r: -r['target'])
    for r in summary_rows:
        print(f'{r["cell"][:45]:<45} {r["n"]:>5} '
                  f'${r["mode"]:>+5.1f} ${r["q25"]:>+5.1f} ${r["q50"]:>+5.1f} '
                  f'${r["q75"]:>+5.1f} ${r["q90"]:>+5.1f} ${r["mean"]:>+5.1f} '
                  f'${r["target"]:>+6.1f}')


if __name__ == '__main__':
    main()

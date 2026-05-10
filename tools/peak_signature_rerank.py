"""Re-rank peak signatures by MOVEMENT signature, not raw coalescence.

The original peak_signature_mining.py ranks features by coalescence
(1/(1+CV) at MFE bar). But features that are SATURATED throughout the
trade (e.g., reversion_prob already at 0.99 at entry because NMP gates
on high rprob) score top-coalescence trivially — they're stable, not
informative.

A real EXIT signal requires the feature to MOVE meaningfully between
entry and MFE. This tool re-ranks per cell by:

    movement_score = |delta_mean| / (entry_std + eps)

i.e., signal-to-noise of the entry→MFE shift. High movement_score means
the feature consistently changes from entry to peak; that change IS the
exit signal.

Combined with MFE coalescence, the cleanest exit features are ones with:
    HIGH movement_score  (feature shifts a lot)  AND
    LOW mfe_cv           (final value is tight across trades)

Output:
    reports/findings/peak_signatures/per_cell_movement_signatures.csv
    reports/findings/peak_signatures/top_movers_per_cell.json

Usage:
    python tools/peak_signature_rerank.py
    python tools/peak_signature_rerank.py --min-n 50 --top-k 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import FEATURE_NAMES
from training_iso_v2.state import REGIME_VOCAB


EPS = 1e-9


def histmode(v: np.ndarray, n_bins: int = 30) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0: return float('nan')
    if v.size == 1: return float(v[0])
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12: return float(np.median(v))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(v, bins=edges)
    j = int(np.argmax(counts))
    return float((edges[j] + edges[j + 1]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet',
                          default='reports/findings/peak_signatures/trade_mfe_features.parquet')
    ap.add_argument('--out-dir',
                          default='reports/findings/peak_signatures')
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--min-n', type=int, default=30)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f'Loaded per-trade frame: {len(df)} rows × {len(df.columns)} cols')

    feat_cols = [c[len('mfe__'):] for c in df.columns if c.startswith('mfe__')]
    print(f'Feature count: {len(feat_cols)}')

    rows = []
    top_per_cell = {}

    cells = df.groupby(['tier', 'regime_idx', 'direction'])
    for (tier, ridx, direction), sub in tqdm(cells, desc='re-rank cells'):
        if len(sub) < args.min_n:
            continue
        regime = (REGIME_VOCAB[ridx] if 0 <= ridx < len(REGIME_VOCAB)
                       else f'IDX{ridx}')
        cell_key = f'{tier}|{regime}|{direction}'

        feat_stats = []
        for f in feat_cols:
            entry_vals = sub[f'entry__{f}'].values
            mfe_vals   = sub[f'mfe__{f}'].values
            d_vals     = sub[f'delta__{f}'].values
            entry_mask = np.isfinite(entry_vals)
            mfe_mask   = np.isfinite(mfe_vals)
            d_mask     = np.isfinite(d_vals)
            if mfe_mask.sum() < args.min_n:
                continue

            entry_mean = float(np.mean(entry_vals[entry_mask]))
            entry_std  = float(np.std(entry_vals[entry_mask], ddof=0))
            mfe_mean = float(np.mean(mfe_vals[mfe_mask]))
            mfe_std  = float(np.std(mfe_vals[mfe_mask], ddof=0))
            mfe_mode = histmode(mfe_vals[mfe_mask])
            mfe_cv   = (mfe_std / abs(mfe_mean)) if abs(mfe_mean) > EPS else float('nan')
            delta_mean = float(np.mean(d_vals[d_mask])) if d_mask.sum() > 0 else 0.0
            delta_std  = float(np.std(d_vals[d_mask], ddof=0)) if d_mask.sum() > 0 else 0.0
            delta_mode = histmode(d_vals[d_mask]) if d_mask.sum() > 0 else float('nan')

            # MOVEMENT signal-to-noise: |delta| / entry_std
            movement = abs(delta_mean) / (entry_std + EPS)
            # Combined: movement × MFE coalescence  (high movement, tight MFE cluster)
            mfe_coalesc = 1.0 / (1.0 + abs(mfe_cv)) if np.isfinite(mfe_cv) else 0.0
            combined = movement * mfe_coalesc

            row = {
                'cell': cell_key, 'tier': tier, 'regime': regime,
                'direction': direction, 'n': len(sub),
                'feature': f,
                'entry_mean': entry_mean, 'entry_std': entry_std,
                'mfe_mean': mfe_mean, 'mfe_mode': mfe_mode,
                'mfe_std': mfe_std, 'mfe_cv': mfe_cv,
                'delta_mean': delta_mean, 'delta_std': delta_std, 'delta_mode': delta_mode,
                'movement_score': movement,
                'mfe_coalescence': mfe_coalesc,
                'combined_score': combined,
            }
            rows.append(row)
            feat_stats.append(row)

        # Top-K by COMBINED score (movement × MFE coalescence)
        feat_stats.sort(key=lambda r: r['combined_score'], reverse=True)
        top_per_cell[cell_key] = [
            {
                'feature': r['feature'],
                'entry_mean': r['entry_mean'],
                'mfe_mode': r['mfe_mode'],
                'delta_mean': r['delta_mean'],
                'movement_score': r['movement_score'],
                'mfe_cv': r['mfe_cv'],
                'combined_score': r['combined_score'],
                'suggested_exit_thr': r['mfe_mode'],
            }
            for r in feat_stats[:args.top_k]
        ]

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'per_cell_movement_signatures.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f'Per-cell movement-rank CSV -> {csv_path}')

    json_path = os.path.join(args.out_dir, 'top_movers_per_cell.json')
    with open(json_path, 'w') as f:
        json.dump(top_per_cell, f, indent=2)
    print(f'Top-K movers per cell    -> {json_path}')

    # Pretty-print focus cells
    print(f'\n{"=" * 130}')
    print(f'TOP-{args.top_k} MOVEMENT-SIGNATURE FEATURES AT MFE  (per tier x regime x direction)')
    print(f'  movement_score = |Delta_entry_to_MFE| / entry_std    (higher = feature consistently moves)')
    print(f'  combined       = movement * (1/(1+|CV_mfe|))         (higher = moves AND lands tight)')
    print(f'{"=" * 130}')

    focus = [
        'RIDE_CALM|UP_SMOOTH|long', 'RIDE_CALM|DOWN_SMOOTH|short',
        'RIDE_MOMENTUM|UP_SMOOTH|long', 'RIDE_MOMENTUM|DOWN_SMOOTH|short',
        'FADE_AGAINST|FLAT_CHOPPY|short', 'FADE_AGAINST|FLAT_CHOPPY|long',
        'FADE_AGAINST|DOWN_CHOPPY|short',
        'NMP_FADE_RAW|FLAT_CHOPPY|short',
        'NMP_RIDE_RAW|UP_SMOOTH|long', 'NMP_RIDE_RAW|DOWN_SMOOTH|short',
    ]
    for cell in focus:
        if cell not in top_per_cell:
            continue
        n = int(summary_df[summary_df['cell'] == cell]['n'].iloc[0]) if (
            (summary_df['cell'] == cell).any()) else 0
        print(f'\n  [{cell}]  n={n}')
        print(f'    {"feature":<46} {"entry_mean":>12} {"mfe_mode":>11} '
                  f'{"delta":>11} {"mvmt":>7} {"cv_mfe":>7} {"combined":>9}')
        for r in top_per_cell[cell]:
            f = r['feature']
            print(f'    {f[:46]:<46} {r["entry_mean"]:>+12.4f} {r["mfe_mode"]:>+11.4f} '
                      f'{r["delta_mean"]:>+11.4f} {r["movement_score"]:>7.2f} '
                      f'{r["mfe_cv"]:>+7.2f} {r["combined_score"]:>9.3f}')


if __name__ == '__main__':
    main()

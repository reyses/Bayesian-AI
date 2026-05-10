"""Peak-signature mining — derive exit thresholds (and entry signatures)
from feature coalescence at empirical MFE points.

USER PREMISE
------------
Instead of sweeping exit-parameter grids, walk every iso trade, find the
bar where MAX FAVORABLE EXCURSION was hit, and read V2 features AT THAT
BAR. Across the population, if features cluster tightly at MFE points
(low CV), those modal values ARE the exit thresholds — derived directly
from data, no fitting.

WHAT THIS OUTPUTS
-----------------
1. Per-trade record (parquet):
       trade_id, tier, regime, direction, mfe_pnl, realized_pnl,
       time_to_mfe_s, capture_ratio,
       entry_v2_<feat>, mfe_v2_<feat>, delta_v2_<feat>
2. Per (tier × regime × direction) summary (csv):
       For each of 185 V2 features:
         mfe_mean, mfe_mode, mfe_std, mfe_cv,
         delta_mean, delta_mode, delta_cv,
         coalescence_score = 1 / (1 + cv)
3. Per-cell top-N most-coalescent features (json):
       { cell: [(feat_name, mode_at_mfe, cv, suggested_exit_thr), ...] }
       suggested_exit_thr = mode_at_mfe (i.e. exit when feature reaches its
       empirical-modal value at peaks).

ENTRY-SIGNATURE FOLLOWUP (auto seeds)
-------------------------------------
The same data supports the auto-seed analysis: stratify trades by MFE
quartile (top vs bottom). At HIGH-MFE trades, what entry features are
common but ABSENT in low-MFE trades? Those are auto-seed candidates.
This script writes the parquet that powers that analysis; a follow-up
tool will read it and stratify.

USAGE
-----
    python tools/peak_signature_mining.py
    python tools/peak_signature_mining.py --prefix training_iso_v2/output/oos
    python tools/peak_signature_mining.py --top-k 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import FEATURE_NAMES, load_features
from training_iso_v2.state import REGIME_VOCAB
from training_iso_v2.ledger import TICK, TICK_VALUE
from training_iso_v2.regret import _load_5s_ohlcv


def histmode(v: np.ndarray, n_bins: int = 30) -> float:
    """Mode via histogram. Returns NaN on degenerate input."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    if v.size == 1:
        return float(v[0])
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return float(np.median(v))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(v, bins=edges)
    j = int(np.argmax(counts))
    return float((edges[j] + edges[j + 1]) / 2)


def find_mfe(trade, ts_arr: np.ndarray, close_arr: np.ndarray,
                  lookahead_pad: int = 0):
    """Walk from entry to entry+bars_held(+pad); return (mfe_bar_idx, mfe_pnl)."""
    idx = int(np.searchsorted(ts_arr, trade.entry_ts))
    if idx >= len(ts_arr):
        return None, None
    end = min(idx + max(int(trade.bars_held), 1) + 1 + lookahead_pad, len(ts_arr))
    if end - idx < 2:
        return None, None
    closes = close_arr[idx:end]
    if trade.direction == 'long':
        pnl_path = (closes - trade.entry_price) / TICK * TICK_VALUE
    else:
        pnl_path = (trade.entry_price - closes) / TICK * TICK_VALUE
    off = int(np.argmax(pnl_path))
    return idx + off, float(pnl_path[off])


def collect_trade_mfes(prefix: str) -> pd.DataFrame:
    """For each trade in iso pickles, capture entry + MFE feature vectors."""
    pkls = sorted(glob.glob(f'{prefix}_*.pkl'))
    pkls = [p for p in pkls if 'regret' not in p and 'summary' not in p]
    if not pkls:
        raise FileNotFoundError(f'No tier pickles at {prefix}_*.pkl')

    all_trades = []
    for path in pkls:
        with open(path, 'rb') as f:
            ts = pickle.load(f)
        all_trades.extend(ts)
    print(f'Loaded {len(all_trades)} trades across {len(pkls)} tier pickles')

    by_day = defaultdict(list)
    for t in all_trades:
        by_day[t.entry_day].append(t)

    rows = []
    for day, trades in tqdm(by_day.items(), desc='peak-mine days'):
        ohlcv = _load_5s_ohlcv(day)
        if ohlcv is None or len(ohlcv) == 0:
            continue
        ts_arr = ohlcv['timestamp'].values.astype(np.int64)
        close_arr = ohlcv['close'].values

        feats = load_features(days=[day])
        if feats.empty:
            continue
        feats_ts = feats['timestamp'].values.astype(np.int64)
        feat_cols = [c for c in FEATURE_NAMES if c in feats.columns]
        feat_mat = feats[feat_cols].values.astype(np.float32)

        for t in trades:
            mfe_bar, mfe_pnl = find_mfe(t, ts_arr, close_arr)
            if mfe_bar is None:
                continue
            mfe_ts = int(ts_arr[mfe_bar])
            fidx = int(np.searchsorted(feats_ts, mfe_ts))
            if fidx >= len(feats_ts):
                continue
            mfe_vec = feat_mat[fidx]

            entry_v2 = (t.entry_v2 if t.entry_v2 is not None and len(t.entry_v2) > 0
                              else np.zeros(len(FEATURE_NAMES), dtype=np.float32))

            row = {
                'day': day,
                'tier': t.entry_tier,
                'regime_idx': int(t.entry_regime_idx),
                'direction': t.direction,
                'entry_ts': float(t.entry_ts),
                'mfe_ts': mfe_ts,
                'time_to_mfe_s': mfe_ts - float(t.entry_ts),
                'mfe_pnl': mfe_pnl,
                'realized_pnl': float(t.pnl),
                'capture_ratio': (float(t.pnl) / mfe_pnl) if mfe_pnl > 0 else 0.0,
                'bars_held': int(t.bars_held),
            }
            for j, fname in enumerate(feat_cols):
                row[f'entry__{fname}'] = float(entry_v2[j])
                row[f'mfe__{fname}'] = float(mfe_vec[j])
                row[f'delta__{fname}'] = float(mfe_vec[j] - entry_v2[j])
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f'Built per-trade MFE-feature dataframe: {len(df)} rows × '
              f'{len(df.columns)} cols')
    return df


def per_cell_signatures(df: pd.DataFrame, top_k: int = 5,
                              min_n: int = 30) -> tuple:
    """For each (tier × regime × direction) cell, compute per-feature stats
    at MFE; return summary dataframe and top-K most-coalescent features."""
    summary_rows = []
    top_per_cell = {}

    feat_cols = [c[len('mfe__'):] for c in df.columns if c.startswith('mfe__')]

    for (tier, ridx, direction), sub in df.groupby(['tier', 'regime_idx', 'direction']):
        if len(sub) < min_n:
            continue
        regime = (REGIME_VOCAB[ridx] if 0 <= ridx < len(REGIME_VOCAB)
                       else f'IDX{ridx}')
        cell_key = f'{tier}|{regime}|{direction}'

        feat_stats = []
        for f in feat_cols:
            mfe_vals = sub[f'mfe__{f}'].values
            d_vals = sub[f'delta__{f}'].values
            mfe_vals = mfe_vals[np.isfinite(mfe_vals)]
            d_vals = d_vals[np.isfinite(d_vals)]
            if mfe_vals.size < min_n:
                continue
            mfe_mean = float(mfe_vals.mean())
            mfe_std = float(mfe_vals.std(ddof=0))
            mfe_mode = histmode(mfe_vals)
            mfe_cv = (mfe_std / abs(mfe_mean)) if abs(mfe_mean) > 1e-9 else float('nan')
            d_mean = float(d_vals.mean()) if d_vals.size > 0 else float('nan')
            d_std = float(d_vals.std(ddof=0)) if d_vals.size > 0 else float('nan')
            d_mode = histmode(d_vals) if d_vals.size > 0 else float('nan')
            coalescence = 1.0 / (1.0 + abs(mfe_cv)) if np.isfinite(mfe_cv) else 0.0

            row = {
                'cell': cell_key, 'tier': tier, 'regime': regime,
                'direction': direction, 'n': len(sub),
                'feature': f,
                'mfe_mean': mfe_mean, 'mfe_mode': mfe_mode,
                'mfe_std': mfe_std, 'mfe_cv': mfe_cv,
                'delta_mean': d_mean, 'delta_mode': d_mode, 'delta_std': d_std,
                'coalescence': coalescence,
            }
            summary_rows.append(row)
            feat_stats.append(row)

        # Top-K by coalescence (highest = features that tightly cluster at MFE)
        feat_stats.sort(key=lambda r: r['coalescence'], reverse=True)
        top_per_cell[cell_key] = [
            {
                'feature': r['feature'],
                'mode_at_mfe': r['mfe_mode'],
                'cv_at_mfe': r['mfe_cv'],
                'coalescence': r['coalescence'],
                'mean_delta_entry_to_mfe': r['delta_mean'],
                'suggested_exit_threshold': r['mfe_mode'],
            }
            for r in feat_stats[:top_k]
        ]

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, top_per_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='training_iso_v2/output/is')
    ap.add_argument('--out-dir',
                          default='reports/findings/peak_signatures')
    ap.add_argument('--top-k', type=int, default=8,
                          help='top-K most-coalescent features per cell')
    ap.add_argument('--min-n', type=int, default=30,
                          help='minimum trades per cell to compute stats')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = collect_trade_mfes(args.prefix)
    if df.empty:
        print('No MFE-feature data; aborting')
        return

    parquet_path = os.path.join(args.out_dir, 'trade_mfe_features.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f'Per-trade frame -> {parquet_path}')

    summary_df, top_per_cell = per_cell_signatures(df, top_k=args.top_k,
                                                                  min_n=args.min_n)
    summary_csv = os.path.join(args.out_dir, 'per_cell_feature_signatures.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f'Per-cell summary -> {summary_csv}')

    json_path = os.path.join(args.out_dir, 'top_features_per_cell.json')
    with open(json_path, 'w') as f:
        json.dump(top_per_cell, f, indent=2)
    print(f'Top-K signatures per cell -> {json_path}')

    # Pretty print: top-K features per cell
    print(f'\n{"=" * 110}')
    print(f'TOP-{args.top_k} MOST-COALESCENT FEATURES AT MFE  (per tier × regime × direction)')
    print(f'  coalescence = 1 / (1 + |CV|)   higher = features cluster tighter at peak')
    print(f'{"=" * 110}')
    for cell, feats in sorted(top_per_cell.items()):
        n = summary_df[summary_df['cell'] == cell]['n'].iloc[0] if (
            (summary_df['cell'] == cell).any()) else 0
        print(f'\n  [{cell}]  n={n}')
        print(f'    {"feature":<42} {"mode@MFE":>12} {"CV":>7} '
                  f'{"coalesc":>8} {"Δentry→MFE":>13}')
        for r in feats:
            print(f'    {r["feature"][:42]:<42} '
                      f'{r["mode_at_mfe"]:>+12.4f} {r["cv_at_mfe"]:>+7.2f} '
                      f'{r["coalescence"]:>8.3f} '
                      f'{r["mean_delta_entry_to_mfe"]:>+13.4f}')

    print(f'\n{">" * 6}  Full per-cell × per-feature stats: {summary_csv}')
    print(f'{">" * 6}  Suggested exit thresholds JSON:    {json_path}')
    print(f'{">" * 6}  Per-trade parquet (for entry auto-seed analysis): {parquet_path}')


if __name__ == '__main__':
    main()

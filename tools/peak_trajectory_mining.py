"""Peak-trajectory mining — capture feature trajectories around MFE.

Extends `peak_signature_mining.py` from sampling ONE bar (MFE bar) to
sampling MULTIPLE offsets relative to MFE. Two exit modes need different
signatures:

    LEAD-TIME EXIT (mode A): detect when peak is FORMING.
        Look at feature pattern at offsets [-20..-1] relative to peak.
        Features that are progressively transitioning toward the peak
        signature give us lead-time to exit AT peak.

    REACTIVE EXIT (mode B): detect when peak has PASSED.
        Look at feature pattern at offsets [+1..+20] relative to peak.
        Features that confirm "peak is gone" let us exit before bleed.

Categories that emerge from the trajectory:
    forming   — monotonic build-up toward peak (useful: lead exit)
    peaked    — spike at offset 0, lower on both sides (useful: lead + lag)
    smooth    — flat across all offsets (NO information; selection bias)
    decaying  — drops after peak (useful: lag exit)

Output:
    reports/findings/peak_trajectory/per_trade_trajectory.parquet
        one row per trade × offset × feature, ~70k × 15 offsets × 185 feats
    reports/findings/peak_trajectory/per_cell_per_feature_trajectory.csv
        per (tier × regime × direction × feature):
            offset, mean, mode, std, cv at that offset
    reports/findings/peak_trajectory/feature_categories.json
        {cell: {forming: [...], peaked: [...], smooth: [...], decaying: [...]}}

Usage:
    python tools/peak_trajectory_mining.py
    python tools/peak_trajectory_mining.py --offsets -10,-5,-3,0,3,5,10
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


DEFAULT_OFFSETS = [-20, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20]


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


def find_mfe(trade, ts_arr: np.ndarray, close_arr: np.ndarray):
    """Walk from entry to entry+bars_held; return (mfe_bar_idx, mfe_pnl)."""
    idx = int(np.searchsorted(ts_arr, trade.entry_ts))
    if idx >= len(ts_arr):
        return None, None
    end = min(idx + max(int(trade.bars_held), 1) + 1, len(ts_arr))
    if end - idx < 2:
        return None, None
    closes = close_arr[idx:end]
    if trade.direction == 'long':
        pnl_path = (closes - trade.entry_price) / TICK * TICK_VALUE
    else:
        pnl_path = (trade.entry_price - closes) / TICK * TICK_VALUE
    off = int(np.argmax(pnl_path))
    return idx + off, float(pnl_path[off])


def collect_trajectories(prefix: str, offsets: list,
                                  max_offset_pad: int = 30) -> pd.DataFrame:
    """For each trade, capture feature vector at MFE±k for k in offsets.
    `max_offset_pad` extends OHLCV walk past actual exit so we can read
    features at peak+k even if k goes past the actual exit bar."""
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
    for day, trades in tqdm(by_day.items(), desc='trajectory days'):
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

            base_row = {
                'day': day,
                'tier': t.entry_tier,
                'regime_idx': int(t.entry_regime_idx),
                'direction': t.direction,
                'entry_ts': float(t.entry_ts),
                'mfe_pnl': mfe_pnl,
                'realized_pnl': float(t.pnl),
                'bars_held': int(t.bars_held),
                'mfe_bar_offset_from_entry': mfe_bar - int(np.searchsorted(
                    ts_arr, t.entry_ts)),
            }

            for off in offsets:
                target_bar = mfe_bar + off
                if target_bar < 0 or target_bar >= len(ts_arr):
                    continue
                target_ts = int(ts_arr[target_bar])
                fidx = int(np.searchsorted(feats_ts, target_ts))
                if fidx >= len(feats_ts):
                    continue
                # PnL at this offset bar
                cl = float(close_arr[target_bar])
                if t.direction == 'long':
                    bar_pnl = (cl - t.entry_price) / TICK * TICK_VALUE
                else:
                    bar_pnl = (t.entry_price - cl) / TICK * TICK_VALUE

                row = dict(base_row)
                row['offset'] = off
                row['bar_pnl'] = bar_pnl
                fvec = feat_mat[fidx]
                for j, fname in enumerate(feat_cols):
                    row[fname] = float(fvec[j])
                rows.append(row)

    df = pd.DataFrame(rows)
    print(f'Built trajectory dataframe: {len(df)} rows × {len(df.columns)} cols')
    return df


def categorize_features(df: pd.DataFrame, feat_cols: list,
                                min_n: int = 30,
                                form_lead_threshold: float = 0.30,
                                spike_ratio_threshold: float = 0.40,
                                smooth_drift_threshold: float = 0.10) -> dict:
    """Per (cell × feature), classify trajectory shape:

        forming  — value at offset -3 differs from offset -20 by >= form_lead_threshold
                    (in std units of the feature)
        peaked   — |val(0) - mean(val(-3,-2,-1,+1,+2,+3))| / std >= spike_ratio_threshold
        decaying — value at offset +3 differs from offset 0 by >= form_lead_threshold
        smooth   — none of the above (drift across all offsets < smooth_drift_threshold)

    Returns: {cell: {category: [(feature, score), ...]}}
    """
    out = {}
    cells = df.groupby(['tier', 'regime_idx', 'direction'])
    for (tier, ridx, direction), sub in cells:
        if sub['offset'].nunique() < 5:
            continue
        n = sub['offset'].value_counts().min()
        if n < min_n:
            continue
        regime = (REGIME_VOCAB[ridx] if 0 <= ridx < len(REGIME_VOCAB)
                       else f'IDX{ridx}')
        cell_key = f'{tier}|{regime}|{direction}'
        cell_cat = {'forming': [], 'peaked': [], 'decaying': [], 'smooth': []}

        for f in feat_cols:
            # Per-offset means
            per_off = sub.groupby('offset')[f].mean()
            std = sub[f].std()
            if not np.isfinite(std) or std < 1e-9:
                continue
            # Helper: get mean at offset, fallback to nearest
            def m(o):
                if o in per_off.index:
                    return float(per_off.loc[o])
                return float('nan')

            v_neg20 = m(-20)
            v_neg3 = m(-3)
            v_0 = m(0)
            v_pos3 = m(3)

            if not all(np.isfinite([v_neg20, v_neg3, v_0, v_pos3])):
                continue

            # Forming: -3 is meaningfully different from -20
            form_strength = abs(v_neg3 - v_neg20) / std
            # Decaying: +3 is meaningfully different from 0
            decay_strength = abs(v_pos3 - v_0) / std
            # Peaked: 0 spikes vs neighbors
            neighbors = [m(o) for o in (-3, -2, -1, 1, 2, 3) if np.isfinite(m(o))]
            if neighbors:
                neigh_mean = float(np.mean(neighbors))
                peak_strength = abs(v_0 - neigh_mean) / std
            else:
                peak_strength = 0.0
            # Smooth: max-min across offsets / std
            offsets_present = [m(o) for o in [-20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20]
                                    if np.isfinite(m(o))]
            spread = max(offsets_present) - min(offsets_present) if offsets_present else 0.0
            spread_norm = spread / std

            # Classify (multi-label allowed)
            if form_strength >= form_lead_threshold:
                cell_cat['forming'].append((f, form_strength))
            if peak_strength >= spike_ratio_threshold:
                cell_cat['peaked'].append((f, peak_strength))
            if decay_strength >= form_lead_threshold:
                cell_cat['decaying'].append((f, decay_strength))
            if spread_norm < smooth_drift_threshold:
                cell_cat['smooth'].append((f, spread_norm))

        # Sort each category by strength
        for cat in cell_cat:
            cell_cat[cat].sort(key=lambda r: r[1], reverse=(cat != 'smooth'))
        out[cell_key] = cell_cat
    return out


def per_cell_per_feature_summary(df: pd.DataFrame, feat_cols: list,
                                              min_n: int = 30) -> pd.DataFrame:
    rows = []
    cells = df.groupby(['tier', 'regime_idx', 'direction'])
    for (tier, ridx, direction), sub in tqdm(cells, desc='per-cell summary'):
        if sub['offset'].nunique() < 5:
            continue
        n = sub['offset'].value_counts().min()
        if n < min_n:
            continue
        regime = (REGIME_VOCAB[ridx] if 0 <= ridx < len(REGIME_VOCAB)
                       else f'IDX{ridx}')
        for f in feat_cols:
            for off, off_sub in sub.groupby('offset'):
                vals = off_sub[f].values
                vals = vals[np.isfinite(vals)]
                if len(vals) < min_n:
                    continue
                rows.append({
                    'tier': tier, 'regime': regime, 'direction': direction,
                    'feature': f, 'offset': int(off), 'n': len(vals),
                    'mean': float(vals.mean()),
                    'mode': histmode(vals),
                    'std': float(vals.std(ddof=0)),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='training_iso_v2/output/is')
    ap.add_argument('--out-dir',
                          default='reports/findings/peak_trajectory')
    ap.add_argument('--offsets', type=str, default=None,
                          help='Comma-sep list (default: -20,-15,-10,-5,-3,-2,-1,0,1,2,3,5,10,15,20)')
    ap.add_argument('--min-n', type=int, default=30)
    args = ap.parse_args()

    if args.offsets:
        offsets = [int(x.strip()) for x in args.offsets.split(',')]
    else:
        offsets = DEFAULT_OFFSETS

    os.makedirs(args.out_dir, exist_ok=True)
    df = collect_trajectories(args.prefix, offsets)
    if df.empty:
        print('!!! No trajectory data; aborting')
        sys.exit(1)

    parquet_path = os.path.join(args.out_dir, 'per_trade_trajectory.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f'Per-trade trajectory frame -> {parquet_path}')

    # Identify feature columns (everything that isn't metadata)
    metadata_cols = {'day', 'tier', 'regime_idx', 'direction', 'entry_ts',
                            'mfe_pnl', 'realized_pnl', 'bars_held',
                            'mfe_bar_offset_from_entry', 'offset', 'bar_pnl'}
    feat_cols = [c for c in df.columns if c not in metadata_cols]

    summary = per_cell_per_feature_summary(df, feat_cols, args.min_n)
    summary_path = os.path.join(args.out_dir,
                                              'per_cell_per_feature_trajectory.csv')
    summary.to_csv(summary_path, index=False)
    print(f'Per-cell trajectory summary -> {summary_path}')

    cat = categorize_features(df, feat_cols, args.min_n)
    cat_path = os.path.join(args.out_dir, 'feature_categories.json')
    with open(cat_path, 'w') as f:
        # Convert tuples to lists for JSON
        ser = {cell: {k: [[name, float(score)] for name, score in lst]
                            for k, lst in v.items()}
                  for cell, v in cat.items()}
        json.dump(ser, f, indent=2)
    print(f'Feature categories per cell -> {cat_path}')

    # Pretty-print summary
    print(f'\n{"=" * 90}')
    print(f'TRAJECTORY-CATEGORY SUMMARY  (top features per category, focus cells)')
    print(f'{"=" * 90}')
    focus = [
        'RIDE_MOMENTUM|UP_SMOOTH|long', 'RIDE_MOMENTUM|DOWN_SMOOTH|short',
        'RIDE_CALM|UP_SMOOTH|long',
        'FADE_AGAINST|FLAT_CHOPPY|short', 'FADE_AGAINST|FLAT_CHOPPY|long',
        'NMP_FADE_RAW|FLAT_CHOPPY|short',
    ]
    for cell in focus:
        if cell not in cat:
            continue
        c = cat[cell]
        print(f'\n  [{cell}]')
        for category in ('forming', 'peaked', 'decaying'):
            top = c[category][:5]
            if not top:
                print(f'    {category:<10}: (none)')
                continue
            line = ', '.join(f'{name}({score:.2f})' for name, score in top)
            print(f'    {category:<10}: {line}')
        sm = c['smooth']
        if sm:
            print(f'    smooth (info-less, top 5): '
                      f'{", ".join(name for name, _ in sm[:5])}')


if __name__ == '__main__':
    main()

"""Build RUN_B2T — the TILED counterpart of RUN_B2C, for a clean same-day A/B.

PURPOSE: isolate ONE variable — tiled vs continuous — with everything else identical
(same day, same 5 TF labels, same L1-L3 layers, same production SFE formulas, same
stage-1 pipeline). RUN_B2C = sliding windows from 1s. RUN_B2T = proper non-overlapping
TF bars, step-filled causally onto the 1s grid (last-CLOSED-bar rule, no lookahead).

If RUN_B2C tiers more cleanly than RUN_B2T at the SAME stage-1 parameter cap (which
saturates both, so it cancels for the comparison), that is evidence FOR the continuous
representation. If they're equivalent, tiling was never the problem.

CAUSAL step-fill: a TF bar with bar_ts=b CLOSES at b+tf_sec; at 1s timestamp ts it may
only be used once ts >= b+tf_sec. merge_asof(direction='backward') on bar_close_ts
enforces this (matches build_dataset _last_closed_idx semantics).

OUTPUT: DATA/ATLAS/FEATURES_RUN_B2T/<day>.parquet  (dir contains "RUN_B" -> stage-1
loader reads directly). Columns match RUN_B2C naming so stage-1 treats them identically.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core_v2.statistical_field_engine import StatisticalFieldEngine, TF_SECONDS
TF_SECONDS.setdefault('3m', 180)   # L5 period for the non-standard 3m label

# (label, tf_seconds, n_base_bars) — MUST match build_run_b2c.py windows.
WINDOWS = [
    ('5s',   5,  9),
    ('15s',  15, 12),
    ('1m',   60, 15),
    ('3m',   180, 12),
    ('5m',   300, 9),
]


def aggregate_bars(df1s: pd.DataFrame, tf_sec: int) -> pd.DataFrame:
    """1s OHLCV -> proper non-overlapping TF bars (floor grouping). Returns bar_ts + OHLCV."""
    ts = df1s['timestamp'].to_numpy(np.int64)
    bar_ts = (ts // tf_sec) * tf_sec
    g = df1s.assign(bar_ts=bar_ts).groupby('bar_ts', sort=True)
    bars = pd.DataFrame({
        'open':   g['open'].first(),
        'high':   g['high'].max(),
        'low':    g['low'].min(),
        'close':  g['close'].last(),
        'volume': g['volume'].sum(),
    }).reset_index()
    return bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--full', action='store_true', help='add L4(NMP) + L5(dist) -> FEATURES_RUN_B2TF; default L1-L3 -> FEATURES_RUN_B2T')
    ap.add_argument('--atlas_root', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS")))
    args = ap.parse_args()

    src = os.path.join(args.atlas_root, '1s', f'{args.day}.parquet')
    if not os.path.exists(src):
        print(f"[FATAL] 1s OHLCV not found: {src}")
        return
    df = pd.read_parquet(src).reset_index(drop=True)
    n = len(df)
    base_ts = df[['timestamp']].copy().sort_values('timestamp')
    print(f"[B2T] {args.day}: {n} 1s rows (tiled counterpart of B2C)")

    sfe = StatisticalFieldEngine()
    parts = [df[['timestamp']].copy(), sfe.compute_L0(df)]

    for label, tf_sec, n_base in WINDOWS:
        bars = aggregate_bars(df, tf_sec)
        if len(bars) <= n_base:
            print(f"[B2T]   {label}: too few bars ({len(bars)}), skipping.")
            continue
        l3 = sfe.compute_L3(bars, label, N=n_base)
        flist = [sfe.compute_L1(bars, label), sfe.compute_L2(bars, label, N=n_base), l3]
        if args.full:
            z_se = l3[f'L3_{label}_z_se_{n_base}'].to_numpy()
            flist.append(sfe.compute_L4_NMP(bars, label, z_se=z_se))   # L4 = NMP lambda/vr/z21 per TF bar
        feats = pd.concat(flist, axis=1)
        feats['bar_close_ts'] = (bars['bar_ts'] + tf_sec).to_numpy()   # causal: usable only after close
        feats = feats.sort_values('bar_close_ts')
        merged = pd.merge_asof(base_ts, feats, left_on='timestamp', right_on='bar_close_ts',
                               direction='backward').drop(columns=['bar_close_ts']).set_index(base_ts.index).sort_index()
        parts.append(merged.drop(columns=['timestamp']).reset_index(drop=True))
        if args.full:
            # L5 = within-bar 1s-close distribution (touch/dwell/wick), step-filled
            ld = sfe.compute_L5_ldist(df, label)
            ld['bar_close_ts'] = ld['bar_ts'] + tf_sec
            ld = ld.sort_values('bar_close_ts')
            m5 = pd.merge_asof(base_ts, ld.drop(columns=['bar_ts']), left_on='timestamp',
                               right_on='bar_close_ts', direction='backward').set_index(base_ts.index).sort_index()
            parts.append(m5.drop(columns=['timestamp', 'bar_close_ts']).reset_index(drop=True))
        print(f"[B2T]   {label}: {len(bars)} bars (tf={tf_sec}s, N={n_base}) -> {'L1-L5' if args.full else 'L1-L3'} step-filled")

    out_df = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    out_dir = os.path.join(args.atlas_root, 'FEATURES_RUN_B2TF' if args.full else 'FEATURES_RUN_B2T')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.day}.parquet')
    out_df.to_parquet(out_path)
    nan_frac = float(out_df.drop(columns=['timestamp']).isnull().any(axis=1).mean())
    print(f"[B2T] wrote {out_path}  shape={out_df.shape}  rows-with-any-NaN={nan_frac:.1%}")


if __name__ == '__main__':
    main()

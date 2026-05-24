"""Build the entry-timing classifier dataset.

For every 1m close in IS and OOS days, label whether it's a "golden entry
moment" — defined as an oracle bar (from the daisy chain) whose MFE exceeds
a threshold (default $200).

Positive class:  1m bars containing a daisy-oracle bar with mfe_dollars > THR
Negative class:  every other 1m bar in session (massive — the user's
                 "exponentially larger" set)

Output: parquet with V2 features at each 1m close + is_golden + oracle_dir
        (the oracle's direction call for positives, NaN for negatives).

The classifier downstream uses (V2 features) -> P(golden); when high, the
direction classifier picks LONG/SHORT.

CRITICAL: features are causal (V2 at that 1m close); label is forward-looking
(whether that bar BECAME a high-MFE move). This is standard supervised
learning, not lookahead.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


V2_ROOT = Path('DATA/ATLAS/FEATURES_5s_v2')
LAYER_DIRS = [
    'L1_5s', 'L1_15s', 'L1_1m', 'L1_5m', 'L1_15m', 'L1_1h', 'L1_4h', 'L1_1D',
    'L2_5s', 'L2_15s', 'L2_1m', 'L2_5m', 'L2_15m', 'L2_1h', 'L2_4h', 'L2_1D',
    'L3_5s', 'L3_15s', 'L3_1m', 'L3_5m', 'L3_15m', 'L3_1h', 'L3_4h', 'L3_1D',
]
TF_S = 5
BARS_PER_1M = 12   # 12 × 5s = 60s


def load_v2_at_1m(day: str) -> pd.DataFrame:
    """Return DataFrame: timestamp + 184 V2 cols, one row per 1m close
    within the day (sampled at every 12th 5s bar starting from the day's
    first bar)."""
    frames = []
    for layer in LAYER_DIRS:
        p = V2_ROOT / layer / f'{day}.parquet'
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        df['timestamp'] = df['timestamp'].astype('int64')
        df = df.sort_values('timestamp').reset_index(drop=True)
        frames.append(df.set_index('timestamp'))
    if not frames:
        return None
    merged = pd.concat(frames, axis=1).reset_index()
    # Filter to TRUE 1m closes: timestamp divisible by 60.
    # (V2 data has irregular sub-5s timestamps; iloc[::12] won't give clean
    # 1m closes. Match the ticker's `is_1m_close = (ts % 60) == 0` semantics.)
    merged = merged[merged['timestamp'] % 60 == 0].reset_index(drop=True)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle-csv', required=True,
                    help='daisy_chain CSV (e.g., daisy_chain_IS_full_daisy.csv)')
    ap.add_argument('--mfe-threshold', type=float, default=200.0,
                    help='Mark oracle bars with mfe_dollars > this as golden (default $200)')
    ap.add_argument('--velocity-threshold', type=float, default=None,
                    help='If set, use mfe_velocity > this ($/min) instead of mfe_dollars')
    ap.add_argument('--days', nargs='*', default=None,
                    help='Specific YYYY_MM_DD list; default = all in oracle CSV')
    ap.add_argument('--target', choices=['is', 'oos'], default='is',
                    help='IS = 2025 days, OOS = 2026 days')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load oracle CSV
    oracle = pd.read_csv(args.oracle_csv)
    oracle['session_date_key'] = pd.to_datetime(oracle['session_date']).dt.strftime('%Y_%m_%d')
    if args.velocity_threshold is not None:
        golden_oracle = oracle[oracle['mfe_velocity'] > args.velocity_threshold].copy()
        filter_label = f'velocity > ${args.velocity_threshold:.1f}/min'
    else:
        golden_oracle = oracle[oracle['mfe_dollars'] > args.mfe_threshold].copy()
        filter_label = f'mfe > ${args.mfe_threshold:.0f}'
    print(f'Oracle bars total: {len(oracle)}')
    print(f'Golden ({filter_label}): {len(golden_oracle)} '
          f'({100*len(golden_oracle)/len(oracle):.1f}%)')

    # Build oracle_ts → direction lookup for golden bars
    golden_dir_by_ts = dict(zip(golden_oracle['oracle_ts'].astype(int),
                                 golden_oracle['direction']))
    golden_mfe_by_ts = dict(zip(golden_oracle['oracle_ts'].astype(int),
                                 golden_oracle['mfe_dollars']))

    # Resolve days
    if args.days:
        days = args.days
    else:
        all_days = sorted(set(oracle['session_date_key'].unique()))
        if args.target == 'is':
            days = [d for d in all_days if d.startswith('2025_')]
        else:
            days = [d for d in all_days if d.startswith('2026_')]
        # Also add days that have V2 data but no oracle entry (rare)
    print(f'Processing {len(days)} {args.target.upper()} days')

    all_rows = []
    n_pos_total = 0
    for day in tqdm(days, desc='days'):
        v2 = load_v2_at_1m(day)
        if v2 is None or len(v2) == 0:
            continue
        # Label: golden bar contains an oracle_ts in (t-60, t]
        v2['is_golden'] = 0
        v2['oracle_dir'] = ''
        v2['oracle_mfe'] = 0.0
        for ts, dir_ in golden_dir_by_ts.items():
            # Find the 1m close bar that includes this ts
            # 1m close at second :55 = covers [t-60, t]
            # Pick the smallest 1m timestamp >= ts where bar's start = ts_bar - 60
            mask = (v2['timestamp'] >= ts - 60) & (v2['timestamp'] < ts + 60)
            if mask.any():
                v2.loc[mask, 'is_golden'] = 1
                v2.loc[mask, 'oracle_dir'] = dir_
                v2.loc[mask, 'oracle_mfe'] = golden_mfe_by_ts.get(ts, 0.0)
        v2['day'] = day
        v2['target_split'] = args.target.upper()
        n_pos_today = int(v2['is_golden'].sum())
        n_pos_total += n_pos_today
        all_rows.append(v2)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_parquet(out_path, index=False)
    print(f'\nWrote: {out_path}')
    print(f'  total 1m bars: {len(out_df)}')
    print(f'  golden positives: {n_pos_total}  ({100*n_pos_total/len(out_df):.2f}%)')
    print(f'  imbalance ratio: {(len(out_df)-n_pos_total)/max(n_pos_total,1):.0f}:1')


if __name__ == '__main__':
    main()

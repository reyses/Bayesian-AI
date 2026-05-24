"""Join daisy-chain trades to the full V2 feature stack at the entry bar.

Per user 2026-05-16: re-run L2 direction discrimination on the full ~190
V2 features (not just the ~19 oracle state-vector subset) to test whether
R²=0.35 is the true ceiling or an artifact of incomplete features.

This tool joins each daisy-chain trade row to its V2 features at the
ENTRY BAR ONLY (single bar per trade). The per-bar-during-duration
trajectory version is deferred for L3.

V2 features live at DATA/ATLAS/FEATURES_5s_v2/<layer>/<YYYY_MM_DD>.parquet:
    L1_5s, L1_15s, L1_1m, L1_5m, L1_15m, L1_1h, L1_4h, L1_1D
    L2_5s, L2_15s, L2_1m, L2_5m, L2_15m, L2_1h, L2_4h, L2_1D
    L3_5s, L3_15s, L3_1m, L3_5m, L3_15m, L3_1h, L3_4h, L3_1D
    plus L0 (timestamp + close)

For each layer/TF, we read its daily parquet for each session_date in the
trade set, then asof-merge onto the trade entry_ts.

Output: enriched parquet (smaller, faster than CSV at this column count).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


V2_ROOT = Path('DATA/ATLAS/FEATURES_5s_v2')
LAYER_DIRS = [
    'L1_5s', 'L1_15s', 'L1_1m', 'L1_5m', 'L1_15m', 'L1_1h', 'L1_4h', 'L1_1D',
    'L2_5s', 'L2_15s', 'L2_1m', 'L2_5m', 'L2_15m', 'L2_1h', 'L2_4h', 'L2_1D',
    'L3_5s', 'L3_15s', 'L3_1m', 'L3_5m', 'L3_15m', 'L3_1h', 'L3_4h', 'L3_1D',
]


def load_layer_for_date(layer: str, date_str: str) -> pd.DataFrame | None:
    """Load a single layer's parquet for a date (date_str = YYYY_MM_DD)."""
    p = V2_ROOT / layer / f'{date_str}.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if 'timestamp' not in df.columns:
        return None
    # Normalize timestamp to int seconds
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    df['timestamp'] = df['timestamp'].astype('int64')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='daisy-chain CSV')
    ap.add_argument('--out', default='reports/findings/regret_oracle/daisy_with_v2_features_IS_full.parquet')
    ap.add_argument('--layers', nargs='*', default=LAYER_DIRS,
                    help='Layer dirs to load (default: all)')
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} trades from {args.input}')

    # Each trade's entry_ts (int seconds), per session_date
    df['_entry_ts_int'] = df['oracle_ts'].astype('int64')
    df['_session_date_key'] = (
        pd.to_datetime(df['session_date'])
          .dt.strftime('%Y_%m_%d')
    )

    # Pre-collect unique session-dates we need to load
    dates = sorted(df['_session_date_key'].unique())
    print(f'Unique session dates: {len(dates)}')

    # We also need the prior date for trades that start near session open
    # (Globex sessions can span the calendar boundary). For safety, load
    # both the session_date AND the prior date. We'll merge_asof onto the
    # union.
    dates_with_prior = set(dates)
    for d in dates:
        try:
            pd_date = pd.to_datetime(d, format='%Y_%m_%d')
            prior = (pd_date - pd.Timedelta(days=1)).strftime('%Y_%m_%d')
            dates_with_prior.add(prior)
        except Exception:
            pass
    dates_with_prior = sorted(dates_with_prior)

    # Result starts as the daisy-chain df
    out_df = df.copy()

    for layer in args.layers:
        # Load all needed daily parquets for this layer, concat, sort
        frames = []
        for d in dates_with_prior:
            x = load_layer_for_date(layer, d)
            if x is not None:
                frames.append(x)
        if not frames:
            print(f'  {layer}: NO DATA in any date — skipping')
            continue
        layer_df = pd.concat(frames, ignore_index=True)
        layer_df = layer_df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        # Prefix non-timestamp columns with layer name to avoid collisions
        rename_map = {c: f'{layer}_{c}' if c != 'timestamp' else c for c in layer_df.columns}
        # Actually, the columns are already prefixed (e.g. 'L1_5s_price_velocity').
        # Don't double-prefix. Just check the first non-timestamp column.
        sample_col = [c for c in layer_df.columns if c != 'timestamp'][0] if len(layer_df.columns) > 1 else None
        if sample_col and sample_col.startswith(layer):
            pass    # already prefixed
        else:
            layer_df = layer_df.rename(columns=rename_map)

        # asof-merge each trade to the most-recent layer row at or before
        # the trade's entry_ts
        merged = pd.merge_asof(
            out_df.sort_values('_entry_ts_int'),
            layer_df.rename(columns={'timestamp': '_entry_ts_int'}),
            on='_entry_ts_int',
            direction='backward',
        )
        # Restore original row order
        out_df = merged.sort_values('oracle_idx').reset_index(drop=True)
        new_cols = [c for c in out_df.columns if c.startswith(layer + '_')]
        print(f'  {layer}: joined ({len(new_cols)} feature columns)')

    # Drop helper columns
    out_df = out_df.drop(columns=['_entry_ts_int', '_session_date_key'], errors='ignore')

    # Write
    out_df.to_parquet(out_path, index=False)
    print(f'\nWrote: {out_path}')
    print(f'  rows: {len(out_df)}')
    print(f'  cols: {len(out_df.columns)}')

    # Summary of V2 columns
    v2_cols = [c for c in out_df.columns if any(c.startswith(layer + '_') for layer in args.layers)]
    print(f'  V2 feature columns: {len(v2_cols)}')

    # NaN audit on V2 columns
    if v2_cols:
        v2_only = out_df[v2_cols]
        nans = v2_only.isna().sum().sum()
        total = v2_only.size
        print(f'  V2 NaN rate: {100*nans/total:.2f}% ({nans:,} of {total:,})')
        # Per-column NaN report for any > 10%
        per_col_nan = (v2_only.isna().sum() / len(v2_only)) * 100
        bad_cols = per_col_nan[per_col_nan > 10].sort_values(ascending=False)
        if len(bad_cols) > 0:
            print(f'  Columns with >10% NaN ({len(bad_cols)} total):')
            for c, p in bad_cols.head(10).items():
                print(f'    {c}: {p:.1f}%')


if __name__ == '__main__':
    main()

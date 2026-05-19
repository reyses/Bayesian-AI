"""Convert NinjaTrader 8 CSV exports to parquet, matching existing format.

CSV format (NT8 raw export, e.g. DATA/ATLAS_NT8/1m/MNQ_06-26/2026_05_15.csv):
  timestamp,open,high,low,close,volume
  1778828400,29403,29410.75,29394.5,29408,1874     (1m: ts = TOP of minute)
  1778828400,29403,29408.5,29402.5,29404,279       (1s: ts = top of second)

Existing parquet format (DATA/ATLAS_NT8/1m/2026_04_27.parquet):
  ts=1777273259  (ts % 60 == 59 = END of minute, bar-close convention)
Existing 5s parquet (DATA/ATLAS_NT8/5s/...):
  ts=1773990004  (ts % 5 == 4 = end of 5s window, bar-close convention)

Conversions:
  1m CSV  -> 1m parquet: shift ts by +59 (top-of-min -> bar-close)
  1s CSV  -> 1s parquet: no shift (NT8 1s already uses start-of-second)
  1s CSV  -> 5s parquet: rebin into 5s windows, label by last second (close)

Validates against existing 2026_03_20 NT8 parquet to confirm alignment.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


# 2026-05-18 path reorg: raw CSVs moved to DATA/RAW_NT8/{contract}/{tf}/
# Parquet stays at DATA/ATLAS_NT8/{tf}/ (flat, no contract subdir)
CONTRACT = 'MNQ_06-26'
CSV_BASE_1M = Path(f'DATA/RAW_NT8/{CONTRACT}/1m')
CSV_BASE_1S = Path(f'DATA/RAW_NT8/{CONTRACT}/1s')
OUT_1M  = Path('DATA/ATLAS_NT8/1m')
OUT_1S  = Path('DATA/ATLAS_NT8/1s')
OUT_5S  = Path('DATA/ATLAS_NT8/5s')


def read_nt8_csv(path: Path) -> pd.DataFrame:
    """Read NT8 CSV, strip BOM, return clean DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip('﻿') for c in df.columns]
    expected = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    if not expected.issubset(df.columns):
        raise ValueError(f'{path}: missing cols. Have {list(df.columns)}')
    return df


def convert_1m(csv_path: Path, out_path: Path) -> dict:
    df = read_nt8_csv(csv_path)
    # CSV uses top-of-minute (ts % 60 == 0); existing parquet uses end-of-min (% 60 == 59)
    df['timestamp'] = df['timestamp'].astype('int64') + 59
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype('float64')
    df['volume'] = df['volume'].astype('int64')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    return {'rows': len(df), 'ts_min': int(df['timestamp'].min()),
            'ts_max': int(df['timestamp'].max()),
            'volume': int(df['volume'].sum())}


def convert_1s(csv_path: Path, out_path: Path) -> dict:
    df = read_nt8_csv(csv_path)
    df['timestamp'] = df['timestamp'].astype('int64')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype('float64')
    df['volume'] = df['volume'].astype('int64')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    return {'rows': len(df), 'ts_min': int(df['timestamp'].min()),
            'ts_max': int(df['timestamp'].max()),
            'volume': int(df['volume'].sum())}


def rebin_1s_to_5s(s1_path: Path, out_path: Path) -> dict:
    df1 = pd.read_parquet(s1_path)
    # NT8 5s convention: bar close ts = window_start + 4 (so % 5 == 4)
    df1['bucket'] = (df1['timestamp'] // 5) * 5 + 4
    g = df1.groupby('bucket', sort=True).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index().rename(columns={'bucket': 'timestamp'})
    g['timestamp'] = g['timestamp'].astype('int64')
    g['volume'] = g['volume'].astype('int64')
    g = g[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    g.to_parquet(out_path, index=False)
    return {'rows': len(g), 'ts_min': int(g['timestamp'].min()),
            'ts_max': int(g['timestamp'].max()),
            'volume': int(g['volume'].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='*', default=None,
                    help='Days to convert (YYYY_MM_DD). Default: all CSVs with no parquet.')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verify', action='store_true',
                    help='Compare rebin output against existing parquet for 2026_03_20')
    args = ap.parse_args()

    if args.verify:
        # Sanity check: re-bin existing 2026_03_20 NT8 1s and compare to existing 5s
        print('=== VERIFY rebin alignment with existing parquet ===')
        s1 = pd.read_parquet('DATA/ATLAS_NT8/1s/2026_03_20.parquet')
        s1['bucket'] = (s1['timestamp'] // 5) * 5 + 4
        g = s1.groupby('bucket', sort=True).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
        ).reset_index().rename(columns={'bucket': 'timestamp'})
        existing = pd.read_parquet('DATA/ATLAS_NT8/5s/2026_03_20.parquet')
        print(f'  Rebin from 1s: {len(g)} rows, vol total {int(g["volume"].sum()):,}')
        print(f'  Existing 5s:   {len(existing)} rows, vol total {int(existing["volume"].sum()):,}')
        # Compare on first 5 ts
        merged = g.merge(existing, on='timestamp', how='inner', suffixes=('_rebin', '_exist'))
        print(f'  Matching ts: {len(merged)}')
        if len(merged) > 0:
            for c in ['open', 'high', 'low', 'close', 'volume']:
                diff = (merged[f'{c}_rebin'] - merged[f'{c}_exist']).abs()
                print(f'  {c}: max abs diff = {diff.max():.4f}, mean abs diff = {diff.mean():.4f}')
        return

    # Discover days: CSV present + parquet missing (or --overwrite)
    csv_1m_days = {Path(p).stem for p in CSV_BASE_1M.glob('2026_*.csv')}
    csv_1s_days = {Path(p).stem for p in CSV_BASE_1S.glob('2026_*.csv')}
    candidate_days = sorted(csv_1m_days | csv_1s_days)

    if args.days:
        candidate_days = [d for d in candidate_days if d in args.days]

    if not candidate_days:
        print('No candidate days. Run with --days YYYY_MM_DD ...')
        return

    print(f'Candidate days: {len(candidate_days)}')
    converted_1m = 0
    converted_1s = 0
    converted_5s = 0
    skipped = 0
    summaries = []

    for day in tqdm(candidate_days, desc='convert'):
        # 1m
        src_1m = CSV_BASE_1M / f'{day}.csv'
        dst_1m = OUT_1M / f'{day}.parquet'
        if src_1m.exists() and (not dst_1m.exists() or args.overwrite):
            info_1m = convert_1m(src_1m, dst_1m)
            converted_1m += 1
        else:
            info_1m = None

        # 1s
        src_1s = CSV_BASE_1S / f'{day}.csv'
        dst_1s = OUT_1S / f'{day}.parquet'
        if src_1s.exists() and (not dst_1s.exists() or args.overwrite):
            info_1s = convert_1s(src_1s, dst_1s)
            converted_1s += 1
        else:
            info_1s = None

        # 5s rebin from 1s parquet (if 1s exists)
        dst_5s = OUT_5S / f'{day}.parquet'
        if dst_1s.exists() and (not dst_5s.exists() or args.overwrite):
            info_5s = rebin_1s_to_5s(dst_1s, dst_5s)
            converted_5s += 1
        else:
            info_5s = None

        summaries.append({'day': day,
                          '1m_rows': info_1m['rows'] if info_1m else None,
                          '1s_rows': info_1s['rows'] if info_1s else None,
                          '5s_rows': info_5s['rows'] if info_5s else None,
                          'vol_1s': info_1s['volume'] if info_1s else None,
                          'vol_5s': info_5s['volume'] if info_5s else None})

    print(f'\nConverted: 1m={converted_1m}, 1s={converted_1s}, 5s={converted_5s}')
    sdf = pd.DataFrame(summaries)
    print(sdf.to_string(index=False))


if __name__ == '__main__':
    main()

"""Cross-validate NT8 vs Databento 1m bars on overlap days.

Reads Databento DBN files from DATA/RAW/ and compares OHLCV to my
NT8 1m parquet (which was rebinned from NT8 1s).

NT8 contract: MNQ JUN26 (= MNQM6 in Databento symbol space).

Databento timestamps are top-of-minute (UTC, ts_event). NT8 uses
bar-close convention (ts % 60 == 59). Convert by adding 59 seconds.

Reports per-day:
  - Row counts (Databento MNQM6 vs NT8 1m)
  - OHLC max/mean diff for matched timestamps
  - Volume diff (sum + per-bar)
  - only_databento and only_nt8 timestamps
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import databento as db
from tqdm import tqdm


# Databento ohlcv-1m batches (in date order)
DATABENTO_BATCHES = [
    'DATA/RAW/GLBX-20260402-DD6HDFKMA9',   # 2025-01-01 to 2026-03-31 1m
    'DATA/RAW/GLBX-20260413-W9W8AKTMAK',   # 2026-03-02 to 2026-04-12 1m (more recent)
]


# Front-month MNQ contract per month (which NT8 file `MNQ_06-26` tracks)
NT8_SYMBOL = 'MNQM6'   # June 2026 contract


def find_dbn_file(day: str) -> str | None:
    """Find Databento 1m DBN file for given YYYY_MM_DD day."""
    db_date = day.replace('_', '')   # 2026_03_20 -> 20260320
    for batch in DATABENTO_BATCHES:
        candidates = glob.glob(f'{batch}/glbx-mdp3-{db_date}.ohlcv-1m.dbn.zst')
        if candidates:
            return candidates[0]
    return None


def read_databento_1m(fpath: str, symbol: str = NT8_SYMBOL) -> pd.DataFrame:
    """Read Databento 1m DBN, filter to given symbol, normalize.

    Databento ts_event = top-of-minute UTC. NT8 = bar-close convention
    (ts % 60 == 59). Shift +59 to align. Both are real UTC (not local).
    Session windows differ:
      - Databento: full CME globex session, 00:00 UTC to 23:59 UTC
      - NT8: 07:00 UTC to 06:59 UTC next day (different "trading day" cutoff)
    So we expect ~17h overlap on intersection days, where matched bars
    should be IDENTICAL.
    """
    data = db.DBNStore.from_file(fpath)
    df = data.to_df().reset_index()
    df = df[df['symbol'] == symbol].copy()
    if len(df) == 0:
        return df
    df['timestamp'] = (df['ts_event'].astype('int64') // 10**9) + 59
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['volume'] = df['volume'].astype('int64')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compare_day(day: str) -> dict:
    """Compare Databento 1m vs NT8 1m in the OVERLAP WINDOW only.

    Each source covers a different 24h slice. We compare only minutes
    that exist in both. Outside the overlap window, each has its own
    rows, which is expected.
    """
    dbn_file = find_dbn_file(day)
    nt8_file = Path(f'DATA/ATLAS_NT8/1m/{day}.parquet')

    result = {'day': day, 'dbn_file': dbn_file, 'nt8_file': str(nt8_file)}

    if dbn_file is None:
        result['error'] = 'no Databento file'
        return result
    if not nt8_file.exists():
        result['error'] = 'no NT8 1m parquet'
        return result

    df_dbn = read_databento_1m(dbn_file)
    df_nt8 = pd.read_parquet(nt8_file).sort_values('timestamp').reset_index(drop=True)

    result['dbn_rows'] = len(df_dbn)
    result['nt8_rows'] = len(df_nt8)
    result['dbn_vol_total'] = int(df_dbn['volume'].sum()) if len(df_dbn) else 0
    result['nt8_vol_total'] = int(df_nt8['volume'].sum())

    if len(df_dbn) == 0:
        result['error'] = 'no Databento rows for MNQM6 symbol'
        return result

    # Overlap window = intersection of both ts ranges
    ts_min = max(int(df_dbn['timestamp'].min()), int(df_nt8['timestamp'].min()))
    ts_max = min(int(df_dbn['timestamp'].max()), int(df_nt8['timestamp'].max()))
    dbn_ov = df_dbn[(df_dbn['timestamp'] >= ts_min) & (df_dbn['timestamp'] <= ts_max)]
    nt8_ov = df_nt8[(df_nt8['timestamp'] >= ts_min) & (df_nt8['timestamp'] <= ts_max)]
    result['overlap_window_min'] = ts_min
    result['overlap_window_max'] = ts_max
    result['dbn_overlap_rows'] = len(dbn_ov)
    result['nt8_overlap_rows'] = len(nt8_ov)

    merged = dbn_ov.merge(nt8_ov, on='timestamp', how='outer',
                           suffixes=('_dbn', '_nt8'), indicator=True)
    result['matched'] = int((merged['_merge'] == 'both').sum())
    result['only_dbn'] = int((merged['_merge'] == 'left_only').sum())
    result['only_nt8'] = int((merged['_merge'] == 'right_only').sum())

    if result['matched'] > 0:
        both = merged[merged['_merge'] == 'both']
        for c in ['open', 'high', 'low', 'close', 'volume']:
            d = (both[f'{c}_dbn'] - both[f'{c}_nt8']).abs()
            result[f'{c}_max_diff'] = float(d.max())
            result[f'{c}_mean_diff'] = float(d.mean())
            result[f'{c}_nonzero'] = int((d > 0).sum())

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='*', default=None,
                    help='Days YYYY_MM_DD. Default: all overlap days.')
    ap.add_argument('--out',
                    default='reports/findings/drs/nt8_vs_databento_1m.txt')
    args = ap.parse_args()

    if args.days:
        days = args.days
    else:
        # Find overlap: days that have both Databento + NT8 1m
        nt8_days = {p.stem for p in Path('DATA/ATLAS_NT8/1m').glob('2026_*.parquet')}
        days = sorted(nt8_days)

    print(f'Comparing {len(days)} day(s) ... (NT8 symbol = {NT8_SYMBOL})')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out(f'NT8 vs Databento 1m cross-validation  (symbol: {NT8_SYMBOL})')
    out('=' * 100)
    out(f'NT8 1m: rebinned from 1s, bar-close convention (ts%60==59)')
    out(f'Databento 1m: ohlcv-1m schema, ts_event top-of-min UTC, +59s to align')
    out('')
    out(f'{"day":<12}  {"dbn":>5}  {"nt8":>5}  '
        f'{"overlap":>7}  {"matched":>7}  '
        f'{"O_diff":>6}  {"H_diff":>6}  {"L_diff":>6}  {"C_diff":>6}  '
        f'{"O_nz":>5}  {"C_nz":>5}  {"vol_dbn":>10}  {"vol_nt8":>10}')

    results = []
    for day in tqdm(days, desc='days'):
        r = compare_day(day)
        results.append(r)
        if 'error' in r:
            out(f'{day:<12}  ERR: {r["error"]}')
            continue
        out(f'{day:<12}  '
            f'{r["dbn_rows"]:>5}  {r["nt8_rows"]:>5}  '
            f'{r.get("dbn_overlap_rows", 0):>7}  '
            f'{r["matched"]:>7}  '
            f'{r.get("open_max_diff", 0):>6.2f}  '
            f'{r.get("high_max_diff", 0):>6.2f}  '
            f'{r.get("low_max_diff", 0):>6.2f}  '
            f'{r.get("close_max_diff", 0):>6.2f}  '
            f'{r.get("open_nonzero", 0):>5}  '
            f'{r.get("close_nonzero", 0):>5}  '
            f'{r["dbn_vol_total"]:>10,}  '
            f'{r["nt8_vol_total"]:>10,}')

    out('')
    out('=== Aggregate ===')
    valid = [r for r in results if 'error' not in r and r.get('matched', 0) > 0]
    if valid:
        total_matched = sum(r['matched'] for r in valid)
        total_only_dbn = sum(r['only_dbn'] for r in valid)
        total_only_nt8 = sum(r['only_nt8'] for r in valid)
        max_close = max(r.get('close_max_diff', 0) for r in valid)
        max_vol = max(r.get('volume_max_diff', 0) for r in valid)
        out(f'Days compared: {len(valid)} / {len(days)}')
        out(f'Total matched bars: {total_matched:,}')
        out(f'Only in Databento: {total_only_dbn}')
        out(f'Only in NT8: {total_only_nt8}')
        out(f'Max close diff across all days: ${max_close:.2f}')
        out(f'Max volume diff: {max_vol:.0f}')
        # Total volume ratios
        total_vol_dbn = sum(r['dbn_vol_total'] for r in valid)
        total_vol_nt8 = sum(r['nt8_vol_total'] for r in valid)
        if total_vol_nt8 > 0:
            out(f'Total volume ratio dbn/nt8: {total_vol_dbn/total_vol_nt8:.4f}')
        # Per-day flag for trouble
        anomalies = [r for r in valid
                       if r.get('close_max_diff', 0) > 1.0 or
                          abs(r['dbn_vol_total'] - r['nt8_vol_total']) / max(r['nt8_vol_total'], 1) > 0.05]
        if anomalies:
            out(f'\nAnomalies (close diff > $1 OR vol diff > 5%):')
            for a in anomalies:
                out(f'  {a["day"]}: close_max=${a.get("close_max_diff", 0):.2f}  '
                    f'vol_dbn={a["dbn_vol_total"]:,}  vol_nt8={a["nt8_vol_total"]:,}  '
                    f'ratio={a["dbn_vol_total"]/max(a["nt8_vol_total"], 1):.3f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()

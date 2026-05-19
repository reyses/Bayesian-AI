"""Rebuild NT8 OOS dataset from 1s parquets to all timeframes (5s..1D).

For each day in DATA/ATLAS_NT8/1s/, generate aligned parquets in:
  5s, 15s, 30s, 1m, 5m, 15m, 30m, 1h, 4h, 1D

Alignment per existing NT8 convention (inferred from overlap days):
  5s   ts % 5    == 4
  15s  ts % 15   == 14
  30s  ts % 30   == 29
  1m   ts % 60   == 59
  5m   ts % 300  == 299
  15m  ts % 900  == 899
  30m  ts % 1800 == 1799
  1h   ts % 3600 == varies per day (3540 most, 2640 some) -- session-relative offset
  4h   ts % 14400 == 14399
  1D   ts % 86400 == 75599 (end-of-session)

The 1h offset varies day-to-day in existing NT8 files. We use the
"per-day natural alignment from 1s data" approach: bar close = last 1s
bar within the period, which produces the standard period-1 offset.
For 1h this may DIFFER from existing NT8 (which may follow session-
relative offsets). We document the diff during validation.

Validation: for each overlap day (2026-03-20 to 2026-04-26), rebin from
1s and compare to existing NT8 parquet. Report per-TF diff stats.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SRC_1S = Path('DATA/ATLAS_NT8/1s')
DST_BASE = Path('DATA/ATLAS_NT8')

# (target_tf, period_seconds, label_offset_from_window_start)
# Standard "bar close" convention: offset = period - 1
TF_SPECS = [
    ('5s',  5,     4),
    ('15s', 15,    14),
    ('30s', 30,    29),
    ('1m',  60,    59),
    ('5m',  300,   299),
    ('15m', 900,   899),
    ('30m', 1800,  1799),
    ('1h',  3600,  3599),    # std offset (will compare against existing 3540/2640)
    ('4h',  14400, 14399),
    ('1D',  86400, 86399),   # std offset (will compare against existing 75599)
]


def rebin_1s(df1s: pd.DataFrame, period: int, label_offset: int) -> pd.DataFrame:
    """Rebin 1s to target period with given label offset.

    bucket = floor((ts - label_offset) / period) * period + label_offset
    Equivalent to: floor(ts / period) * period + label_offset, with
    adjustment when ts < label_offset within period.
    """
    ts = df1s['timestamp'].values.astype(np.int64)
    # Bucket: each 1s row's bar-close label
    # First, compute the "natural" bucket start as floor(ts/period)*period
    # Then assign each row to the bucket whose label >= ts (next close)
    # Simplified: bucket_label = ((ts // period) * period) + label_offset
    # But if label_offset < period-1, some rows shift up:
    # e.g. period=3600 offset=3540: ts=1773993541 should go to bucket 1773997140 not 1773993540
    # Use: bucket = (ts + period - 1 - label_offset) // period * period + label_offset
    bucket = ((ts + period - 1 - label_offset) // period) * period + label_offset
    df = df1s.copy()
    df['bucket'] = bucket
    g = df.groupby('bucket', sort=True).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index().rename(columns={'bucket': 'timestamp'})
    g['timestamp'] = g['timestamp'].astype('int64')
    g['volume'] = g['volume'].astype('int64')
    g = g[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return g


def compare_two(df_new: pd.DataFrame, df_existing: pd.DataFrame, label: str) -> dict:
    """Compare rebinned vs existing parquet. Returns diff stats."""
    n_new, n_old = len(df_new), len(df_existing)
    merged = df_new.merge(df_existing, on='timestamp', how='outer',
                            suffixes=('_new', '_old'), indicator=True)
    only_new = (merged['_merge'] == 'left_only').sum()
    only_old = (merged['_merge'] == 'right_only').sum()
    matched = (merged['_merge'] == 'both').sum()

    diffs = {}
    if matched > 0:
        both = merged[merged['_merge'] == 'both']
        for c in ['open', 'high', 'low', 'close', 'volume']:
            d = (both[f'{c}_new'] - both[f'{c}_old']).abs()
            diffs[c] = (float(d.max()), float(d.mean()))

    return {
        'tf': label,
        'n_new': n_new, 'n_old': n_old, 'matched': int(matched),
        'only_new': int(only_new), 'only_old': int(only_old),
        'diffs': diffs,
    }


def process_day(day: str, mode: str = 'validate') -> dict:
    """Rebin one day from 1s to all target TFs.

    mode:
      'validate'    -> compare against existing parquet, do not write
      'write-new'   -> write only if file missing; skip if existing
      'overwrite'   -> always write
    """
    src_path = SRC_1S / f'{day}.parquet'
    if not src_path.exists():
        return {'day': day, 'error': 'no 1s source'}

    df1s = pd.read_parquet(src_path).sort_values('timestamp').reset_index(drop=True)
    day_report = {'day': day, 'src_rows': len(df1s), 'tf_results': []}

    for tf, period, offset in TF_SPECS:
        rebin = rebin_1s(df1s, period, offset)

        dst_path = DST_BASE / tf / f'{day}.parquet'
        if mode == 'validate':
            if dst_path.exists():
                df_existing = pd.read_parquet(dst_path).sort_values('timestamp').reset_index(drop=True)
                diff = compare_two(rebin, df_existing, tf)
                day_report['tf_results'].append(diff)
            else:
                day_report['tf_results'].append({'tf': tf, 'missing_existing': True,
                                                    'n_new': len(rebin)})
        elif mode == 'write-new':
            if dst_path.exists():
                day_report['tf_results'].append({'tf': tf, 'skipped': str(dst_path)})
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                rebin.to_parquet(dst_path, index=False)
                day_report['tf_results'].append({'tf': tf, 'wrote': str(dst_path),
                                                    'n_rows': len(rebin)})
        elif mode == 'overwrite':
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            rebin.to_parquet(dst_path, index=False)
            day_report['tf_results'].append({'tf': tf, 'wrote': str(dst_path),
                                                'n_rows': len(rebin)})
        else:
            raise ValueError(f'bad mode: {mode}')

    return day_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='*', default=None,
                    help='Days YYYY_MM_DD. Default: all 1s parquets.')
    ap.add_argument('--mode', choices=['validate', 'write-new', 'overwrite'],
                    default='validate',
                    help='validate (compare only), write-new (skip existing), '
                         'overwrite (always write)')
    ap.add_argument('--validate', action='store_true',
                    help='[DEPRECATED] same as --mode validate')
    ap.add_argument('--report-out',
                    default='reports/findings/drs/nt8_rebin_validation.txt')
    args = ap.parse_args()
    if args.validate:
        args.mode = 'validate'

    if args.days:
        days = sorted(args.days)
    else:
        days = sorted([p.stem for p in SRC_1S.glob('2026_*.parquet')])

    print(f'Mode: {args.mode}')
    print(f'Days: {len(days)}')
    if days:
        print(f'  First: {days[0]}   Last: {days[-1]}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out(f'NT8 REBUILD FROM 1s -- mode={args.mode}')
    out('=' * 100)
    out(f'Days: {len(days)}')
    out(f'Target TFs: {[s[0] for s in TF_SPECS]}')
    out('')

    all_reports = []
    for day in tqdm(days, desc='days'):
        rep = process_day(day, mode=args.mode)
        all_reports.append(rep)

    # Count write actions
    if args.mode != 'validate':
        n_wrote = sum(1 for rep in all_reports for r in rep.get('tf_results', [])
                       if 'wrote' in r)
        n_skipped = sum(1 for rep in all_reports for r in rep.get('tf_results', [])
                         if 'skipped' in r)
        out(f'Wrote: {n_wrote} parquet files')
        out(f'Skipped existing: {n_skipped}')

    if args.mode == 'validate':
        out('=== Per-TF validation summary ===')
        out(f'{"TF":>4}  {"days":>4}  '
            f'{"matched_avg":>11}  {"only_new_avg":>12}  {"only_old_avg":>12}  '
            f'{"max_OHLC_diff":>13}  {"max_vol_diff":>12}')
        for tf, period, offset in TF_SPECS:
            day_results = [r for rep in all_reports for r in rep.get('tf_results', [])
                            if r.get('tf') == tf and 'diffs' in r]
            if not day_results:
                continue
            matched_avg = np.mean([r['matched'] for r in day_results])
            only_new_avg = np.mean([r['only_new'] for r in day_results])
            only_old_avg = np.mean([r['only_old'] for r in day_results])
            ohlc_diffs = [r['diffs'].get(c, (0,0))[0]
                            for c in ['open','high','low','close']
                            for r in day_results if r.get('diffs')]
            max_ohlc = max(ohlc_diffs) if ohlc_diffs else 0.0
            vol_diffs = [r['diffs'].get('volume', (0,0))[0]
                            for r in day_results if r.get('diffs')]
            max_vol = max(vol_diffs) if vol_diffs else 0.0
            out(f'{tf:>4}  {len(day_results):>4}  '
                f'{matched_avg:>11.0f}  {only_new_avg:>12.1f}  {only_old_avg:>12.1f}  '
                f'{max_ohlc:>13.4f}  {max_vol:>12.0f}')

        out('')
        out('--- Per-day per-TF detail (only TFs with mismatches) ---')
        for rep in all_reports:
            mismatches = [r for r in rep.get('tf_results', [])
                            if 'diffs' in r and (r['only_new'] > 0 or r['only_old'] > 0
                                or any(d[0] > 0 for d in r['diffs'].values()))]
            if not mismatches:
                continue
            out(f'\n{rep["day"]} (src 1s rows: {rep["src_rows"]:,}):')
            for r in mismatches:
                out(f'  {r["tf"]}: n_new={r["n_new"]} n_old={r["n_old"]} '
                    f'matched={r["matched"]} only_new={r["only_new"]} '
                    f'only_old={r["only_old"]}')
                for c, (mx, mn) in r['diffs'].items():
                    if mx > 0:
                        out(f'    {c}: max_diff={mx:.4f}  mean_diff={mn:.4f}')

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.report_out}')


if __name__ == '__main__':
    main()

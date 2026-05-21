"""Mock-replay a week of trading days through engine_v2 (L5 zigzag engine).

Validates that the L5 pipeline works end-to-end on multiple consecutive
days. Each day runs as a separate engine_v2 subprocess so session state
is fully isolated. Per-day output files (ledger / trades) land at
`reports/live/v2_{ledger,trades,nt8_trades}_YYYY_MM_DD.csv`.

After all days run, aggregates per-day P&L and compares to OOS forward
pass expectation for the same days.

Usage:
    python tools/mock_week_runner.py
    python tools/mock_week_runner.py --days 2026_05_11 2026_05_12 ...
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_DAYS = [
    '2026_05_11', '2026_05_12', '2026_05_13',
    '2026_05_14', '2026_05_15',
]


def run_one_day(day: str, timeout_s: int = 600) -> dict:
    """Spawn engine_v2 in mock mode for one day. Returns timing + path info."""
    cmd = [
        sys.executable, '-m', 'live.engine_v2',
        '--mock', '--mock-day', day,
        '--headless',
        '--skip-check', '--skip-build',
    ]
    print(f'\n[{day}]  Launching: {" ".join(cmd[2:])}')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
    elapsed = time.time() - t0
    return {
        'day': day,
        'returncode': result.returncode,
        'elapsed_s': elapsed,
        'stdout_tail': '\n'.join(result.stdout.strip().split('\n')[-30:]),
        'stderr_tail': '\n'.join(result.stderr.strip().split('\n')[-10:]) if result.stderr else '',
    }


def collect_per_day_pnl(day: str) -> dict:
    """Read the per-day output files and extract P&L summary."""
    out = {'day': day}
    trades_path = Path(f'reports/live/v2_trades_{day}.csv')
    nt8_path = Path(f'reports/live/nt8_trades_{day}.csv')

    if trades_path.exists():
        try:
            tdf = pd.read_csv(trades_path)
            out['v2_trades_rows'] = len(tdf)
            # Look for FILL_ENTRY / EXIT events
            entries = tdf[tdf['type'].str.startswith('FILL_ENTRY', na=False)]
            exits = tdf[tdf['type'].astype(str).str.startswith('FILL_EXIT', na=False)]
            out['v2_entries'] = len(entries)
            out['v2_exits'] = len(exits)
            out['v2_total_pnl'] = float(tdf['pnl'].sum()) if 'pnl' in tdf else 0.0
        except Exception as e:
            out['v2_trades_error'] = str(e)
    else:
        out['v2_trades_rows'] = 0

    if nt8_path.exists():
        try:
            ndf = pd.read_csv(nt8_path)
            out['nt8_trades_rows'] = len(ndf)
            out['nt8_total_pnl'] = float(ndf['pnl'].sum()) if 'pnl' in ndf else 0.0
        except Exception as e:
            out['nt8_trades_error'] = str(e)
    else:
        out['nt8_trades_rows'] = 0

    return out


def lookup_oos_expectation(day: str) -> dict:
    """Pull the OOS forward pass's prediction for this day from per_day CSV."""
    p = Path('reports/findings/regret_oracle/charts/2026-05-19_1contract_per_day.csv')
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    r = df[df['day'] == day]
    if r.empty:
        return {}
    return {
        'oos_flat':       float(r['FLAT 1c'].iloc[0]),
        'oos_phase1':     float(r['B7 + B9 + B10 (Phase-1 1c)'].iloc[0]),
        'oos_b10_mode':   str(r['b10_mode'].iloc[0]),
        'oos_n_legs':     int(r['n_legs'].iloc[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='+', default=DEFAULT_DAYS)
    ap.add_argument('--timeout', type=int, default=600,
                    help='Per-day timeout in seconds')
    ap.add_argument('--skip-run', action='store_true',
                    help='Skip subprocess runs; just collect existing output files')
    args = ap.parse_args()

    print('=' * 80)
    print('MOCK WEEK RUNNER  (L5 zigzag engine)')
    print(f'Days: {args.days}')
    print('=' * 80)

    run_results = []
    for day in args.days:
        if args.skip_run:
            run_results.append({'day': day, 'returncode': 0, 'elapsed_s': 0,
                                  'stdout_tail': '(skipped)', 'stderr_tail': ''})
            continue
        r = run_one_day(day, args.timeout)
        status = 'OK' if r['returncode'] == 0 else f'FAIL ({r["returncode"]})'
        print(f'  [{day}]  {status}  ({r["elapsed_s"]:.0f}s)')
        if r['returncode'] != 0:
            print(f'    stderr_tail:\n{r["stderr_tail"]}')
            print(f'    stdout_tail:\n{r["stdout_tail"]}')
        run_results.append(r)

    print()
    print('=' * 80)
    print('PER-DAY RESULTS')
    print('=' * 80)
    print(f'{"day":<14}  {"status":<8}  {"mock_$":>10}  '
          f'{"oos_phase1":>10}  {"oos_flat":>10}  {"n_trades":>8}  '
          f'{"mode":<10}')
    print('-' * 80)

    rows = []
    for r in run_results:
        day = r['day']
        pnl = collect_per_day_pnl(day)
        oos = lookup_oos_expectation(day)
        rows.append({**r, **pnl, **oos})
        status = 'OK' if r['returncode'] == 0 else 'FAIL'
        print(f'{day:<14}  {status:<8}  '
              f'${pnl.get("nt8_total_pnl", pnl.get("v2_total_pnl", 0)):>+9,.0f}  '
              f'${oos.get("oos_phase1", 0):>+9,.0f}  '
              f'${oos.get("oos_flat", 0):>+9,.0f}  '
              f'{pnl.get("v2_entries", 0):>8}  '
              f'{oos.get("oos_b10_mode", "?"):<10}')

    print()
    df = pd.DataFrame(rows)
    if not df.empty and 'nt8_total_pnl' in df.columns:
        mock_total = float(df['nt8_total_pnl'].fillna(df.get('v2_total_pnl', 0)).sum())
        oos_phase1_total = float(df.get('oos_phase1', pd.Series()).sum())
        oos_flat_total = float(df.get('oos_flat', pd.Series()).sum())
        print(f'5-DAY TOTAL  mock=${mock_total:+,.0f}  '
              f'oos_phase1=${oos_phase1_total:+,.0f}  '
              f'oos_flat=${oos_flat_total:+,.0f}')
        if oos_phase1_total != 0:
            ratio = mock_total / oos_phase1_total
            print(f'mock / oos_phase1 ratio: {ratio:.2f}x  '
                  f'(1.00 = perfect match; <0.5 or >2.0 = investigate)')

    # Save summary
    out_csv = Path('reports/live/mock_week_summary.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')


if __name__ == '__main__':
    main()

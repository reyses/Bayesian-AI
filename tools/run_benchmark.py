"""
Benchmark Runner — Daisy-chain IS + OOS forward passes + snapshot results

Usage:
    python tools/run_benchmark.py                      # IS (ATLAS) then OOS (ATLAS_OOS)
    python tools/run_benchmark.py --tag "trail_act_fix" # tag the snapshot for easy ID
    python tools/run_benchmark.py --is-only             # just IS forward pass
    python tools/run_benchmark.py --oos-only            # just OOS forward pass
    python tools/run_benchmark.py --history             # print benchmark history table

Each run saves a timestamped snapshot to reports/benchmarks/ so you can track
improvements across code changes without losing old results.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARKS_DIR = os.path.join(ROOT, 'reports', 'benchmarks')
HISTORY_CSV = os.path.join(BENCHMARKS_DIR, 'history.csv')

HISTORY_FIELDS = [
    'timestamp', 'tag', 'phase',
    'trades', 'win_rate', 'total_pnl', 'avg_pnl',
    'correct_dir_pct', 'wrong_dir_pct',
    'trail_stop_n', 'trail_stop_wr',
    'structural_break_n', 'structural_break_wr',
    'avg_hold_bars',
]


def _parse_trade_log(path):
    """Extract summary metrics from an oracle_trade_log.csv."""
    import pandas as pd
    import numpy as np

    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None

    # Normalize column name (older logs use 'pnl', newer use 'actual_pnl')
    if 'actual_pnl' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['actual_pnl']

    # Hold time
    if 'entry_time' in df.columns and 'exit_time' in df.columns:
        bars = ((df['exit_time'] - df['entry_time']) / 15).clip(lower=1)
    elif 'duration' in df.columns:
        bars = (df['duration'] / 15).clip(lower=1)
    else:
        bars = pd.Series([float('nan')])

    metrics = {
        'trades': len(df),
        'win_rate': f"{(df.pnl > 0).mean():.3f}",
        'total_pnl': f"{df.pnl.sum():.2f}",
        'avg_pnl': f"{df.pnl.mean():.2f}",
        'avg_hold_bars': f"{bars.mean():.1f}",
    }

    # Direction
    if 'oracle_label' in df.columns:
        metrics['correct_dir_pct'] = f"{(df.oracle_label > 0).mean():.3f}"
        metrics['wrong_dir_pct'] = f"{(df.oracle_label < 0).mean():.3f}"
    else:
        metrics['correct_dir_pct'] = ''
        metrics['wrong_dir_pct'] = ''

    # Exit reasons
    for reason in ['trail_stop', 'structural_break']:
        sub = df[df.exit_reason == reason] if 'exit_reason' in df.columns else pd.DataFrame()
        metrics[f'{reason}_n'] = len(sub)
        metrics[f'{reason}_wr'] = f"{(sub.pnl > 0).mean():.3f}" if len(sub) > 0 else ''

    return metrics


def _snapshot(tag, phase, reports_dir):
    """Save trade log + metrics to timestamped benchmark folder."""
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    log_path = os.path.join(reports_dir, 'oracle_trade_log.csv')
    metrics = _parse_trade_log(log_path)
    if metrics is None:
        print(f"  [skip] No trade log found at {log_path}")
        return

    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    snap_dir = os.path.join(BENCHMARKS_DIR, f"{ts}_{phase}_{tag or 'run'}")
    os.makedirs(snap_dir, exist_ok=True)

    # Copy key files
    for fname in ('oracle_trade_log.csv', 'phase4_report.txt',
                  'signal_log.csv', 'pid_oracle_log.csv'):
        src = os.path.join(reports_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, snap_dir)

    # Append to history CSV
    row = {'timestamp': ts, 'tag': tag or '', 'phase': phase}
    row.update(metrics)

    write_header = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Snapshot saved: {snap_dir}")
    print(f"  {phase}: {metrics['trades']} trades, "
          f"WR={float(metrics['win_rate']):.1%}, "
          f"PnL=${float(metrics['total_pnl']):,.0f}")


def _run_forward(args_list, label):
    """Run orchestrator with given args."""
    cmd = [sys.executable, '-m', 'training.orchestrator'] + args_list
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  FAILED: {label} (exit code {result.returncode})")
        return False
    return True


def print_history():
    """Print benchmark history table."""
    if not os.path.exists(HISTORY_CSV):
        print("No benchmark history yet. Run a benchmark first.")
        return

    with open(HISTORY_CSV, 'r') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No benchmark history yet.")
        return

    print(f"\n{'=' * 100}")
    print(f"  BENCHMARK HISTORY ({len(rows)} runs)")
    print(f"{'=' * 100}")
    print(f"  {'Timestamp':<20s} {'Tag':<18s} {'Phase':<5s} {'Trades':>7s} "
          f"{'WR':>6s} {'PnL':>10s} {'Avg':>8s} {'Dir%':>6s} {'Hold':>6s}")
    print(f"  {'-' * 95}")

    for r in rows:
        wr = f"{float(r['win_rate']):.1%}" if r.get('win_rate') else '  -'
        pnl = f"${float(r['total_pnl']):>8,.0f}" if r.get('total_pnl') else '  -'
        avg = f"${float(r['avg_pnl']):>6.1f}" if r.get('avg_pnl') else '  -'
        dir_pct = f"{float(r['correct_dir_pct']):.0%}" if r.get('correct_dir_pct') else '  -'
        hold = f"{float(r['avg_hold_bars']):.0f}b" if r.get('avg_hold_bars') else '  -'
        print(f"  {r['timestamp']:<20s} {r.get('tag', ''):<18s} {r['phase']:<5s} "
              f"{r['trades']:>7s} {wr:>6s} {pnl:>10s} {avg:>8s} {dir_pct:>6s} {hold:>6s}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark runner: IS + OOS daisy chain")
    parser.add_argument('--tag', default=None,
                        help="Label this run (e.g. 'trail_act_fix', 'v2_exits')")
    parser.add_argument('--is-only', action='store_true', help="Run IS forward pass only")
    parser.add_argument('--oos-only', action='store_true', help="Run OOS forward pass only")
    parser.add_argument('--history', action='store_true', help="Print benchmark history and exit")
    parser.add_argument('--no-dashboard', action='store_true', default=True,
                        help="Disable dashboard (default: True for benchmarks)")
    parser.add_argument('--extra', nargs='*', default=[],
                        help="Extra args to pass to orchestrator (e.g. --extra --min-tier 2)")
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.history:
        print_history()
        return

    is_dir = os.path.join(ROOT, 'reports', 'is')
    oos_dir = os.path.join(ROOT, 'reports', 'oos')

    dashboard_args = ['--no-dashboard'] if args.no_dashboard else []
    extra = args.extra or []

    # -- Phase 1: IS forward pass ------------------------------------------
    if not args.oos_only:
        ok = _run_forward(
            ['--forward-pass'] + dashboard_args + extra,
            "IS Forward Pass (full ATLAS)"
        )
        if ok:
            _snapshot(args.tag, 'IS', is_dir)
        else:
            print("  IS run failed, skipping OOS")
            sys.exit(1)

    # -- Phase 2: OOS forward pass -----------------------------------------
    if not args.is_only:
        ok = _run_forward(
            ['--forward-pass', '--oos', '--forward-data', 'DATA/ATLAS_OOS']
            + dashboard_args + extra,
            "OOS Forward Pass (ATLAS_OOS)"
        )
        if ok:
            _snapshot(args.tag, 'OOS', oos_dir)

    # -- Summary -----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    if not args.oos_only and os.path.exists(os.path.join(is_dir, 'oracle_trade_log.csv')):
        m = _parse_trade_log(os.path.join(is_dir, 'oracle_trade_log.csv'))
        if m:
            print(f"  IS:  {m['trades']} trades, WR={float(m['win_rate']):.1%}, PnL=${float(m['total_pnl']):,.0f}")
    if not args.is_only and os.path.exists(os.path.join(oos_dir, 'oracle_trade_log.csv')):
        m = _parse_trade_log(os.path.join(oos_dir, 'oracle_trade_log.csv'))
        if m:
            print(f"  OOS: {m['trades']} trades, WR={float(m['win_rate']):.1%}, PnL=${float(m['total_pnl']):,.0f}")
    print(f"\n  View history: python tools/run_benchmark.py --history")
    print(f"  Deep dive:    python tools/analyze_exits.py --file reports/is/oracle_trade_log.csv")
    print(f"  Compare:      python tools/compare_oos_runs.py")


if __name__ == '__main__':
    main()

"""
Baseline Runner — physics-only forward pass with clean report.

Runs BlendedEngine (no CNN) on IS + OOS + OOS-NT8.
Prints a single pasteable report at the end.

Usage:
    python training/run_baseline.py              # full run
    python training/run_baseline.py --oos-only   # skip IS
    python training/run_baseline.py --atlas DATA/ATLAS_NT8  # NT8 data only
    python training/run_baseline.py --skip-sundays          # filter Sunday sessions
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine
from collections import Counter

# Holiday-adjacent dates with same thin-market signature as Sundays
# These are empirically identified losers with <25 trades and high velocity
HOLIDAY_DATES = {'2025_12_18', '2025_01_02'}  # pre-Christmas, post-New Year


def is_sunday(day_name: str) -> bool:
    """Check if YYYY_MM_DD falls on a Sunday."""
    try:
        dt = datetime.strptime(day_name, '%Y_%m_%d')
        return dt.weekday() == 6  # 6 = Sunday
    except ValueError:
        return False


def filter_thin_market_days(feat_files: list, skip_sundays: bool) -> list:
    """Remove Sunday and holiday-adjacent sessions (thin liquidity)."""
    if not skip_sundays:
        return feat_files
    filtered = []
    skipped = []
    for f in feat_files:
        day_name = os.path.basename(f).replace('.parquet', '')
        if is_sunday(day_name) or day_name in HOLIDAY_DATES:
            skipped.append(day_name)
        else:
            filtered.append(f)
    if skipped:
        print(f'  Skipped {len(skipped)} thin-market days '
              f'({sum(1 for s in skipped if is_sunday(s))} Sun + '
              f'{sum(1 for s in skipped if s in HOLIDAY_DATES)} holiday)')
    return filtered

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
FEATURES_NT8 = 'DATA/FEATURES_NT8_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
ATLAS_NT8_1M = 'DATA/ATLAS_NT8/1m'


def run_forward(feat_files, price_dir, label=''):
    """Run physics-only forward pass. Returns (results, trades)."""
    engine = BlendedEngine(use_cnn=False)
    all_results = []
    all_trades = []

    for fpath in tqdm(feat_files, desc=f'  {label}', unit='day', leave=False):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(price_dir, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades:
            t['day'] = day_name
        all_trades.extend(engine.get_full_trades())

        all_results.append({
            'day': day_name,
            'trades': len(engine.trades),
            'pnl': engine.daily_pnl,
        })

    return all_results, all_trades


REPORT_PATH = 'reports/baseline_report.txt'
DAILY_CSV_PATH = 'reports/baseline_daily.csv'


class ReportWriter:
    """Dual output: prints to stdout AND captures to file."""

    def __init__(self):
        self._lines = []

    def p(self, text=''):
        print(text)
        self._lines.append(text)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self._lines) + '\n')
        print(f'  Report saved → {path}')


def build_report(datasets, tier_trades=None, skip_sundays=False):
    """Build full report to stdout + file. Returns ReportWriter."""
    rw = ReportWriter()

    rw.p()
    rw.p('=' * 70)
    tag = ' [skip-sundays]' if skip_sundays else ''
    rw.p(f'BASELINE REPORT (physics only, no CNN){tag}')
    rw.p(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    rw.p('=' * 70)
    rw.p()

    # ── Summary table ──────────────────────────────────────────────────
    rw.p(f'{"Dataset":<16} {"$/day":>8} {"Trades":>8} {"WinDays":>12} {"Days":>6}')
    rw.p('-' * 56)

    for label, results in datasets.items():
        if not results:
            continue
        n = len(results)
        total_pnl = sum(r['pnl'] for r in results)
        per_day = total_pnl / max(n, 1)
        total_trades = sum(r['trades'] for r in results)
        win_days = sum(1 for r in results if r['pnl'] > 0)
        wr_pct = win_days / n * 100 if n else 0
        rw.p(f'{label:<16} {per_day:>+8,.0f} {total_trades:>8,} '
             f'{win_days:>4}/{n:<4} ({wr_pct:.0f}%) {n:>4}')

    # ── Tier breakdown ─────────────────────────────────────────────────
    if tier_trades:
        primary = [t for t in tier_trades if not str(t.get('exit_reason', '')).startswith('chain_')]
        chains = [t for t in tier_trades if str(t.get('exit_reason', '')).startswith('chain_')]
        n_days = len(set(t.get('day', '') for t in tier_trades)) or 1

        rw.p()
        rw.p('PRIMARY TRADES:')
        rw.p(f'{"Tier":<20} {"Trades":>7} {"WR":>5} {"$/tr":>8} {"$/day":>8}')
        rw.p('-' * 52)

        p_tiers = Counter(t.get('entry_tier', '?') for t in primary)
        for tier, count in sorted(p_tiers.items(), key=lambda x: -sum(
                t['pnl'] for t in primary if t.get('entry_tier') == x[0])):
            sub = [t for t in primary if t.get('entry_tier') == tier]
            wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            avg = np.mean([t['pnl'] for t in sub])
            pd_val = sum(t['pnl'] for t in sub) / n_days
            rw.p(f'{str(tier):<20} {count:>7} {wr:>4.0f}% {avg:>8.1f} {pd_val:>+8.0f}')
        rw.p(f'{"TOTAL PRIMARY":<20} {len(primary):>7} {"":>5} {"":>8} '
             f'{sum(t["pnl"] for t in primary)/n_days:>+8.0f}')

        if chains:
            rw.p()
            rw.p('CHAIN TRADES:')
            rw.p(f'{"Tier":<20} {"Trades":>7} {"WR":>5} {"$/tr":>8} {"$/day":>8}')
            rw.p('-' * 52)

            c_tiers = Counter(t.get('entry_tier', '?') for t in chains)
            for tier, count in sorted(c_tiers.items(), key=lambda x: -sum(
                    t['pnl'] for t in chains if t.get('entry_tier') == x[0])):
                sub = [t for t in chains if t.get('entry_tier') == tier]
                wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
                avg = np.mean([t['pnl'] for t in sub])
                pd_val = sum(t['pnl'] for t in sub) / n_days
                rw.p(f'{str(tier):<20} {count:>7} {wr:>4.0f}% {avg:>8.1f} {pd_val:>+8.0f}')
            rw.p(f'{"TOTAL CHAIN":<20} {len(chains):>7} {"":>5} {"":>8} '
                 f'{sum(t["pnl"] for t in chains)/n_days:>+8.0f}')

    # ── Distribution stats per dataset ─────────────────────────────────
    rw.p()
    for label, results in datasets.items():
        if not results or len(results) < 5:
            continue
        pnls = np.array([r['pnl'] for r in results])
        rw.p(f'{label} DISTRIBUTION:')
        rw.p(f'  Mean:   ${np.mean(pnls):>+,.0f}')
        rw.p(f'  Median: ${np.median(pnls):>+,.0f}')
        # Mode: bin to $250 buckets, find most common
        bucket_size = 250
        bins = (pnls / bucket_size).astype(int) * bucket_size
        mode_bucket = Counter(bins).most_common(1)[0]
        rw.p(f'  Mode:   ${mode_bucket[0]:>+,} bucket ({mode_bucket[1]} days)')
        p5, p25, p75, p95 = np.percentile(pnls, [5, 25, 75, 95])
        rw.p(f'  P5/P25/P75/P95: ${p5:>+,.0f} / ${p25:>+,.0f} / ${p75:>+,.0f} / ${p95:>+,.0f}')
        worst = min(results, key=lambda r: r['pnl'])
        best = max(results, key=lambda r: r['pnl'])
        rw.p(f'  Worst:  ${worst["pnl"]:>+,.0f} ({worst["day"]})')
        rw.p(f'  Best:   ${best["pnl"]:>+,.0f} ({best["day"]})')
        rw.p()

    # ── Day-of-week breakdown per dataset ──────────────────────────────
    rw.p('DAY-OF-WEEK ANALYSIS:')
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for label, results in datasets.items():
        if not results:
            continue
        rw.p(f'  {label}:')
        rw.p(f'  {"Day":<12} {"Days":>5} {"Win":>5} {"WR":>5} {"$/day":>8} {"Losing":>8}')
        rw.p(f'  {"-"*50}')
        by_dow = {}
        for r in results:
            try:
                dt = datetime.strptime(r['day'], '%Y_%m_%d')
                dow = dow_names[dt.weekday()]
            except ValueError:
                dow = '?'
            by_dow.setdefault(dow, []).append(r)
        for dow in dow_names:
            if dow not in by_dow:
                continue
            days = by_dow[dow]
            n = len(days)
            wins = sum(1 for d in days if d['pnl'] > 0)
            wr = wins / n * 100 if n else 0
            avg = np.mean([d['pnl'] for d in days])
            losers = [d for d in days if d['pnl'] < 0]
            loss_str = f'{len(losers)}' if losers else '-'
            rw.p(f'  {dow:<12} {n:>5} {wins:>5} {wr:>4.0f}% {avg:>+8,.0f} {loss_str:>8}')
        rw.p()

    # ── Losing days detail ─────────────────────────────────────────────
    rw.p('LOSING DAYS:')
    for label, results in datasets.items():
        losers = sorted([r for r in results if r['pnl'] < 0], key=lambda r: r['pnl'])
        if not losers:
            rw.p(f'  {label}: NONE')
            continue
        rw.p(f'  {label} ({len(losers)} losing):')
        rw.p(f'  {"Day":<14} {"DOW":<10} {"PnL":>8} {"Trades":>7}')
        rw.p(f'  {"-"*42}')
        for r in losers:
            try:
                dt = datetime.strptime(r['day'], '%Y_%m_%d')
                dow = dow_names[dt.weekday()]
            except ValueError:
                dow = '?'
            rw.p(f'  {r["day"]:<14} {dow:<10} {r["pnl"]:>+8,.0f} {r["trades"]:>7}')
        # Count Sundays
        sun_count = sum(1 for r in losers
                        if datetime.strptime(r['day'], '%Y_%m_%d').weekday() == 6)
        rw.p(f'  → {sun_count}/{len(losers)} are Sundays')
        rw.p()

    rw.p('=' * 70)
    return rw


def save_daily_csv(datasets, out_path=DAILY_CSV_PATH):
    """Save per-day PnL to CSV for downstream analysis."""
    rows = []
    for label, results in datasets.items():
        for r in results:
            try:
                dow = datetime.strptime(r['day'], '%Y_%m_%d').strftime('%A')
            except ValueError:
                dow = '?'
            rows.append({
                'dataset': label,
                'day': r['day'],
                'pnl': round(r['pnl'], 2),
                'trades': r['trades'],
                'dow': dow,
                'is_sunday': dow == 'Sunday',
            })
    if rows:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f'  Saved {len(rows)} daily results → {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Baseline Runner')
    parser.add_argument('--oos-only', action='store_true')
    parser.add_argument('--atlas', type=str, default=None)
    parser.add_argument('--skip-sundays', action='store_true',
                        help='Skip Sunday + holiday-adjacent thin-market sessions')
    args = parser.parse_args()

    t0 = time.perf_counter()
    datasets = {}

    if args.skip_sundays:
        print('  [FILTER] Skipping Sundays + holiday-adjacent thin-market days')

    if args.atlas:
        # Single dataset mode
        atlas_name = os.path.basename(args.atlas.rstrip('/'))
        feat_name = atlas_name.replace('ATLAS', 'FEATURES')
        feat_dir = os.path.join('DATA', f'{feat_name}_5s')
        price_dir = os.path.join(args.atlas, '1m')

        feat_files = sorted(glob.glob(os.path.join(feat_dir, '*.parquet')))
        feat_files = filter_thin_market_days(feat_files, args.skip_sundays)
        if not feat_files:
            print(f'No features in {feat_dir}/')
            return

        results, trades = run_forward(feat_files, price_dir, atlas_name)
        datasets[atlas_name] = results
        rw = build_report(datasets, trades, skip_sundays=args.skip_sundays)

    else:
        # Full run: IS + OOS + OOS-NT8
        all_trades = []

        if not args.oos_only:
            is_files = sorted(f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                              if '2025_' in os.path.basename(f))
            is_files = filter_thin_market_days(is_files, args.skip_sundays)
            if is_files:
                results, trades = run_forward(is_files, ATLAS_1M, 'IS')
                datasets['IS'] = results
                all_trades.extend(trades)

        oos_files = sorted(f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                           if '2026_' in os.path.basename(f))
        oos_files = filter_thin_market_days(oos_files, args.skip_sundays)
        if oos_files:
            results, trades = run_forward(oos_files, ATLAS_1M, 'OOS')
            datasets['OOS'] = results

        nt8_files = sorted(glob.glob(os.path.join(FEATURES_NT8, '*.parquet')))
        nt8_files = filter_thin_market_days(nt8_files, args.skip_sundays)
        if nt8_files:
            results, trades = run_forward(nt8_files, ATLAS_NT8_1M, 'OOS-NT8')
            datasets['OOS-NT8'] = results

        rw = build_report(datasets, all_trades if all_trades else None,
                          skip_sundays=args.skip_sundays)

    # Save report file + daily CSV
    report_path = REPORT_PATH
    if args.skip_sundays:
        report_path = REPORT_PATH.replace('.txt', '_no_sundays.txt')
    rw.save(report_path)
    save_daily_csv(datasets)

    elapsed = time.perf_counter() - t0
    print(f'  Elapsed: {elapsed:.0f}s')


if __name__ == '__main__':
    main()

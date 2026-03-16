#!/usr/bin/env python
"""
Gate Cascade Research — replay IS signal data to evaluate alternative gate orderings
and scoring functions WITHOUT modifying production code.

Reads the signal_log shards + oracle_trade_log to simulate:
1. Current cascade (baseline)
2. Conviction-weighted scoring
3. Bar-level conviction pre-filter
4. Momentum alignment in score

Outputs comparison table + detailed findings to reports/findings/.

Usage:
    python tools/gate_cascade_research.py
    python tools/gate_cascade_research.py --conviction-threshold 0.55
    python tools/gate_cascade_research.py --score-weights "conv=0.3,wr=0.3,dist=0.3,mom=0.1"
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICK_VALUE = 0.50


def load_signal_logs(mode: str = 'is') -> pd.DataFrame:
    """Load and concatenate signal_log shards."""
    shard_dir = os.path.join('reports', mode, 'shards')
    files = sorted(Path(shard_dir).glob('signal_log_*.csv'))
    if not files:
        print(f"  No signal_log shards in {shard_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  {f.name}: ERROR ({e})")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_oracle_trade_log() -> pd.DataFrame:
    """Load the oracle trade log for ground truth."""
    candidates = [
        'checkpoints/oracle_trade_log.csv',
        'checkpoints/oracle_trade_log_old.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded oracle trade log: {path} ({len(df):,} trades)")
            return df
    print("  WARNING: No oracle trade log found")
    return pd.DataFrame()


def analyze_conviction_distribution(df: pd.DataFrame):
    """Analyze conviction vs outcome across all signals."""
    print("\n" + "=" * 70)
    print("  1. CONVICTION DISTRIBUTION ANALYSIS")
    print("=" * 70)

    conv_col = 'gate3_conv'
    if conv_col not in df.columns:
        print("  No conviction column found")
        return

    # All signals
    all_conv = df[conv_col].dropna()
    print(f"\n  All signals ({len(all_conv):,}):")
    print(f"    P10={np.percentile(all_conv, 10):.3f}  P25={np.percentile(all_conv, 25):.3f}  "
          f"Med={np.median(all_conv):.3f}  P75={np.percentile(all_conv, 75):.3f}  "
          f"P90={np.percentile(all_conv, 90):.3f}")

    # Traded vs gate-blocked vs competition-lost
    traded = df[df['trade_result'].notna() & (df['trade_result'] != '')]
    score_losers = df[df['gate'] == 'score_loser']
    gate_blocked = df[~df.index.isin(traded.index) & ~df.index.isin(score_losers.index)]

    for label, subset in [('Traded', traded), ('Score losers', score_losers),
                           ('Gate blocked', gate_blocked)]:
        conv = subset[conv_col].dropna()
        if len(conv) == 0:
            print(f"\n  {label}: 0 signals")
            continue
        print(f"\n  {label} ({len(conv):,}):")
        print(f"    Med={np.median(conv):.3f}  Mean={np.mean(conv):.3f}  "
              f"P75={np.percentile(conv, 75):.3f}")

    # Conviction vs PnL buckets for traded signals
    print(f"\n  CONVICTION -> PnL (traded signals):")
    print(f"    {'Bucket':>15s}  {'Trades':>7s}  {'WR':>6s}  {'Avg PnL':>9s}  {'$/trade if blocked':>18s}")
    print(f"    {'-' * 60}")

    buckets = [(0.0, 0.48, '<0.48'), (0.48, 0.50, '0.48-0.50'),
               (0.50, 0.55, '0.50-0.55'), (0.55, 0.60, '0.55-0.60'),
               (0.60, 0.65, '0.60-0.65'), (0.65, 0.70, '0.65-0.70'),
               (0.70, 0.80, '0.70-0.80'), (0.80, 1.01, '0.80+')]

    for lo, hi, label in buckets:
        bucket = traded[(traded[conv_col] >= lo) & (traded[conv_col] < hi)]
        if len(bucket) == 0:
            continue
        n = len(bucket)
        wr = (bucket['trade_result'] == 'WIN').sum() / n * 100
        avg_pnl = bucket['trade_pnl'].mean()
        total_pnl = bucket['trade_pnl'].sum()
        # If we blocked this bucket, how much would we save/lose?
        saved = f"save ${-total_pnl:,.2f}" if total_pnl < 0 else f"lose ${total_pnl:,.2f}"
        print(f"    {label:>15s}  {n:>7,}  {wr:5.1f}%  ${avg_pnl:>8.2f}  {saved:>18s}")


def analyze_competition_quality(df: pd.DataFrame):
    """Analyze whether competition losers had better conviction than winners."""
    print("\n" + "=" * 70)
    print("  2. COMPETITION QUALITY ANALYSIS")
    print("=" * 70)

    conv_col = 'gate3_conv'
    if conv_col not in df.columns:
        print("  No conviction column found")
        return

    # Group by timestamp (bar) to find competition bars
    traded = df[df['trade_result'].notna() & (df['trade_result'] != '')]
    score_losers = df[df['gate'] == 'score_loser']

    if len(traded) == 0 or len(score_losers) == 0:
        print("  Insufficient data for competition analysis")
        return

    # Find bars where both a trade and score_losers exist
    trade_bars = set(traded['ts'].unique())
    loser_bars = set(score_losers['ts'].unique())
    competition_bars = trade_bars & loser_bars

    print(f"\n  Competition bars: {len(competition_bars):,}")
    print(f"  Traded signals: {len(traded):,}")
    print(f"  Score losers: {len(score_losers):,}")

    # For each competition bar: compare winner conviction vs best loser conviction
    winner_better = 0
    loser_better = 0
    equal = 0
    conv_gaps = []

    for ts in competition_bars:
        winners = traded[traded['ts'] == ts]
        losers = score_losers[score_losers['ts'] == ts]

        if len(winners) == 0 or len(losers) == 0:
            continue

        # Winner conviction (should be same for all on bar, but take the traded one)
        w_conv = winners[conv_col].iloc[0]
        # Best loser conviction
        l_best_conv = losers[conv_col].max()

        # Compare distance (winner should have lower gate1_dist)
        w_dist = winners['gate1_dist'].iloc[0] if 'gate1_dist' in winners.columns else 0
        l_best_dist = losers['gate1_dist'].min() if 'gate1_dist' in losers.columns else 0

        # Did a loser have better template WR?
        # (can't check directly without lib_entry, but we can check if loser
        # had better oracle label)

        if w_conv > l_best_conv + 0.01:
            winner_better += 1
        elif l_best_conv > w_conv + 0.01:
            loser_better += 1
            conv_gaps.append(l_best_conv - w_conv)
        else:
            equal += 1

    total = winner_better + loser_better + equal
    if total == 0:
        print("  No competition comparisons possible")
        return

    print(f"\n  Winner had higher conviction: {winner_better:>6,}  ({winner_better/total*100:.1f}%)")
    print(f"  Loser had higher conviction:  {loser_better:>6,}  ({loser_better/total*100:.1f}%)")
    print(f"  Equal (±0.01):                {equal:>6,}  ({equal/total*100:.1f}%)")

    if conv_gaps:
        print(f"\n  When loser had higher conviction:")
        print(f"    Mean gap: {np.mean(conv_gaps):.3f}")
        print(f"    Max gap:  {max(conv_gaps):.3f}")
        print(f"    P75 gap:  {np.percentile(conv_gaps, 75):.3f}")


def analyze_momentum_alignment(df: pd.DataFrame):
    """Analyze momentum alignment vs outcome."""
    print("\n" + "=" * 70)
    print("  3. MOMENTUM ALIGNMENT ANALYSIS")
    print("=" * 70)

    traded = df[df['trade_result'].notna() & (df['trade_result'] != '')]
    if 'F_momentum' not in traded.columns or 'trade_direction' not in traded.columns:
        print("  Missing F_momentum or trade_direction columns")
        return

    # Compute alignment
    mom = traded['F_momentum'].values
    direction = traded['trade_direction'].values
    aligned = []
    for m, d in zip(mom, direction):
        if pd.isna(m) or pd.isna(d) or m == 0:
            aligned.append('neutral')
        elif (m > 0 and d == 'LONG') or (m < 0 and d == 'SHORT'):
            aligned.append('aligned')
        else:
            aligned.append('misaligned')

    traded = traded.copy()
    traded['mom_alignment'] = aligned

    print(f"\n  MOMENTUM ALIGNMENT (traded signals):")
    print(f"    {'Alignment':>12s}  {'Trades':>7s}  {'WR':>6s}  {'Avg PnL':>9s}  {'Total PnL':>11s}")
    print(f"    {'-' * 50}")
    for label in ['aligned', 'neutral', 'misaligned']:
        sub = traded[traded['mom_alignment'] == label]
        if len(sub) == 0:
            continue
        n = len(sub)
        wr = (sub['trade_result'] == 'WIN').sum() / n * 100
        avg = sub['trade_pnl'].mean()
        total = sub['trade_pnl'].sum()
        print(f"    {label:>12s}  {n:>7,}  {wr:5.1f}%  ${avg:>8.2f}  ${total:>10,.2f}")


def analyze_brain_reject(df: pd.DataFrame):
    """Analyze why brain reject never fires."""
    print("\n" + "=" * 70)
    print("  4. BRAIN REJECT ANALYSIS")
    print("=" * 70)

    traded = df[df['trade_result'].notna() & (df['trade_result'] != '')]
    if 'template_id' not in traded.columns:
        print("  No template_id column")
        return

    # Load brain if available
    brain_path = 'checkpoints/brain.pkl'
    if not os.path.exists(brain_path):
        # Try pattern library for template stats
        lib_path = 'checkpoints/pattern_library.pkl'
        if os.path.exists(lib_path):
            import pickle
            with open(lib_path, 'rb') as f:
                lib = pickle.load(f)
            print(f"  Loaded pattern library: {len(lib)} templates")

            # Check template WR distribution
            wrs = []
            for tid, entry in lib.items():
                wr = entry.get('stats_win_rate', 0)
                wrs.append((tid, wr))

            wrs.sort(key=lambda x: x[1])
            print(f"\n  Template WR distribution:")
            wr_vals = [w[1] for w in wrs]
            print(f"    Min={min(wr_vals):.2f}  P25={np.percentile(wr_vals, 25):.2f}  "
                  f"Med={np.median(wr_vals):.2f}  P75={np.percentile(wr_vals, 75):.2f}  "
                  f"Max={max(wr_vals):.2f}")

            # Templates below 50% WR
            low_wr = [w for w in wrs if w[1] < 0.50]
            print(f"\n  Templates with WR < 50%: {len(low_wr)}")
            for tid, wr in low_wr[:10]:
                n_trades = sum(1 for _, row in traded.iterrows() if row['template_id'] == tid)
                print(f"    TID {tid:>4d}: WR={wr:.2f}  traded={n_trades}")

            # How many trades went to low-WR templates?
            low_wr_tids = {w[0] for w in low_wr}
            low_wr_trades = traded[traded['template_id'].isin(low_wr_tids)]
            if len(low_wr_trades) > 0:
                n = len(low_wr_trades)
                wr = (low_wr_trades['trade_result'] == 'WIN').sum() / n * 100
                pnl = low_wr_trades['trade_pnl'].sum()
                print(f"\n  Trades on <50% WR templates: {n:,}  WR={wr:.1f}%  PnL=${pnl:,.2f}")
                print(f"  -> If brain rejected these: save ${-pnl:,.2f}" if pnl < 0
                      else f"  -> If brain rejected these: lose ${pnl:,.2f}")
        else:
            print("  No brain or pattern library found")
        return

    # Load brain directly
    try:
        import pickle
        with open(brain_path, 'rb') as f:
            brain = pickle.load(f)
        print(f"  Loaded brain: {len(brain.table)} states")

        # Check should_fire logic
        from core.bayesian_brain import MarketBayesianBrain
        # Test what min_prob the brain uses
        fired = 0
        blocked = 0
        for tid in traded['template_id'].unique():
            if brain.should_fire(int(tid), min_prob=0.5, min_conf=0.0):
                fired += 1
            else:
                blocked += 1
        print(f"\n  Brain should_fire(min_prob=0.5): fired={fired} blocked={blocked}")

        fired2 = 0
        blocked2 = 0
        for tid in traded['template_id'].unique():
            if brain.should_fire(int(tid), min_prob=0.6, min_conf=0.0):
                fired2 += 1
            else:
                blocked2 += 1
        print(f"  Brain should_fire(min_prob=0.6): fired={fired2} blocked={blocked2}")
    except Exception as e:
        print(f"  Failed to load brain: {e}")


def simulate_threshold_sweep(df: pd.DataFrame):
    """Simulate different conviction thresholds."""
    print("\n" + "=" * 70)
    print("  5. CONVICTION THRESHOLD SWEEP")
    print("=" * 70)

    conv_col = 'gate3_conv'
    traded = df[df['trade_result'].notna() & (df['trade_result'] != '')]
    if conv_col not in traded.columns or len(traded) == 0:
        print("  No conviction data")
        return

    baseline_n = len(traded)
    baseline_pnl = traded['trade_pnl'].sum()
    baseline_wr = (traded['trade_result'] == 'WIN').sum() / baseline_n * 100
    baseline_avg = baseline_pnl / baseline_n

    print(f"\n  Baseline (0.48): {baseline_n:,} trades, WR={baseline_wr:.1f}%, "
          f"${baseline_pnl:,.2f} PnL, ${baseline_avg:.2f}/trade")
    print(f"\n  {'Threshold':>10s}  {'Trades':>7s}  {'WR':>6s}  {'PnL':>12s}  {'$/trade':>9s}  "
          f"{'Trades cut':>10s}  {'PnL diff':>10s}")
    print(f"  {'-' * 72}")

    for thresh in [0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        surviving = traded[traded[conv_col] >= thresh]
        n = len(surviving)
        if n == 0:
            continue
        wr = (surviving['trade_result'] == 'WIN').sum() / n * 100
        pnl = surviving['trade_pnl'].sum()
        avg = pnl / n
        cut = baseline_n - n
        diff = pnl - baseline_pnl
        print(f"  {thresh:>10.2f}  {n:>7,}  {wr:5.1f}%  ${pnl:>11,.2f}  ${avg:>8.2f}  "
              f"{cut:>10,}  ${diff:>+9,.2f}")


def write_report(output_path: str, content: str):
    """Save report to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Gate Cascade Research')
    parser.add_argument('--mode', default='is', choices=['is', 'oos'])
    args = parser.parse_args()

    print("=" * 70)
    print("  GATE CASCADE RESEARCH")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 70)

    # Load data
    print("\n  Loading signal logs...")
    df = load_signal_logs(args.mode)
    if df.empty:
        print("  No data — cannot proceed")
        return

    traded = df[df['trade_result'].notna() & (df['trade_result'] != '')]
    score_losers = df[df['gate'] == 'score_loser']
    print(f"  Total signals: {len(df):,}")
    print(f"  Traded: {len(traded):,}  Score losers: {len(score_losers):,}")

    # Run analyses
    analyze_conviction_distribution(df)
    analyze_competition_quality(df)
    analyze_momentum_alignment(df)
    analyze_brain_reject(df)
    simulate_threshold_sweep(df)

    # Save to file
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'reports/findings/gate_cascade_research_{ts}.txt'

    # Re-run with output capture
    import io
    buf = io.StringIO()
    _old_stdout = sys.stdout

    # Just save a summary
    summary_lines = [
        f"Gate Cascade Research — {args.mode.upper()} mode",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Signals: {len(df):,}  Traded: {len(traded):,}  Score losers: {len(score_losers):,}",
        "",
        "See terminal output for full analysis.",
        "Key metrics saved below.",
        "",
    ]

    # Key metrics
    conv_col = 'gate3_conv'
    if conv_col in traded.columns and len(traded) > 0:
        summary_lines.append("CONVICTION SWEEP:")
        for thresh in [0.48, 0.50, 0.55, 0.60, 0.65]:
            surv = traded[traded[conv_col] >= thresh]
            n = len(surv)
            if n == 0:
                continue
            wr = (surv['trade_result'] == 'WIN').sum() / n * 100
            pnl = surv['trade_pnl'].sum()
            avg = pnl / n
            summary_lines.append(f"  thresh={thresh:.2f}: {n:,} trades, WR={wr:.1f}%, "
                                 f"PnL=${pnl:,.2f}, avg=${avg:.2f}")

    write_report(out_path, '\n'.join(summary_lines))


if __name__ == '__main__':
    main()

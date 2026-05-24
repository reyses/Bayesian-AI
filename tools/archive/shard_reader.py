#!/usr/bin/env python
"""
Shard Reader — quick summary of IS/OOS signal log shards during or after a run.

Reads partial results from reports/{mode}/shards/signal_log_*.csv without
waiting for the full forward pass to complete.

Usage:
    python tools/shard_reader.py              # IS shards (default)
    python tools/shard_reader.py --mode oos   # OOS shards
    python tools/shard_reader.py --depth      # per-depth breakdown
    python tools/shard_reader.py --exits      # exit reason breakdown
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

TICK_VALUE = 0.50


def read_shards(mode: str = 'is') -> pd.DataFrame:
    """Load and concatenate all signal log shards for the given mode."""
    shard_dir = os.path.join('reports', mode, 'shards')
    if not os.path.isdir(shard_dir):
        print(f"  No shard directory: {shard_dir}")
        return pd.DataFrame()

    files = sorted(Path(shard_dir).glob('signal_log_*.csv'))
    if not files:
        print(f"  No signal_log shards in {shard_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                df['_shard'] = f.stem
                dfs.append(df)
                print(f"  {f.name}: {len(df):,} trades")
        except Exception as e:
            print(f"  {f.name}: ERROR ({e})")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Filter to traded signals only (non-empty trade_result or trade_pnl != 0)
    if 'trade_result' in df.columns:
        df = df[df['trade_result'].notna() & (df['trade_result'] != '')].copy()
    elif 'trade_pnl' in df.columns:
        df = df[df['trade_pnl'].notna() & (df['trade_pnl'] != 0)].copy()

    print(f"  Traded signals: {len(df):,}")
    return df


def print_summary(df: pd.DataFrame, show_depth: bool = False, show_exits: bool = False):
    """Print trade summary from shard data."""
    n = len(df)
    if n == 0:
        print("  No trades found.")
        return

    # Normalize column names (signal_log uses trade_pnl, oracle_trade_log uses pnl)
    _pnl_col = 'trade_pnl' if 'trade_pnl' in df.columns else 'pnl'
    has_pnl = _pnl_col in df.columns
    pnl = df[_pnl_col].sum() if has_pnl else 0
    wins = (df[_pnl_col] > 0).sum() if has_pnl else 0
    losses = (df[_pnl_col] <= 0).sum() if has_pnl else 0
    wr = wins / n * 100
    avg = pnl / n

    print(f"\n{'=' * 70}")
    print(f"  SHARD SUMMARY ({len(df['_shard'].unique())} shards)")
    print(f"{'=' * 70}")
    print(f"  Trades: {n:,}  |  WR: {wr:.1f}%  |  PnL: ${pnl:,.2f}  |  Avg: ${avg:.2f}/trade")
    print(f"  Wins: {wins:,}  |  Losses: {losses:,}")

    # Duration stats
    if 'hold_bars' in df.columns:
        avg_hold = df['hold_bars'].mean() * 15 / 60  # 15s bars → minutes
        print(f"  Avg hold: {avg_hold:.1f} min")

    # Exit reason breakdown
    if show_exits and 'exit_reason' in df.columns:
        print(f"\n  EXIT REASONS:")
        for reason, group in df.groupby('exit_reason'):
            rn = len(group)
            rpnl = group[_pnl_col].sum() if has_pnl else 0
            rwr = (group[_pnl_col] > 0).sum() / rn * 100 if has_pnl else 0
            print(f"    {reason:25s}  {rn:>5,} trades  WR={rwr:5.1f}%  PnL=${rpnl:>10,.2f}")

    # Per-depth breakdown
    if show_depth and 'depth' in df.columns:
        print(f"\n  PER-DEPTH BREAKDOWN:")
        print(f"    {'Depth':>8s}  {'Trades':>7s}  {'WR':>6s}  {'PnL':>12s}  {'Avg':>8s}")
        print(f"    {'-' * 50}")
        for d in sorted(df['depth'].unique()):
            dd = df[df['depth'] == d]
            dn = len(dd)
            dwr = (dd[_pnl_col] > 0).sum() / dn * 100 if has_pnl else 0
            dpnl = dd[_pnl_col].sum() if has_pnl else 0
            davg = dpnl / dn
            print(f"    depth {d:>2.0f}  {dn:>7,}  {dwr:5.1f}%  ${dpnl:>11,.2f}  ${davg:>7.2f}")

    # Per-shard breakdown (quarterly)
    print(f"\n  PER-SHARD BREAKDOWN:")
    for shard in sorted(df['_shard'].unique()):
        sd = df[df['_shard'] == shard]
        sn = len(sd)
        swr = (sd[_pnl_col] > 0).sum() / sn * 100 if has_pnl else 0
        spnl = sd[_pnl_col].sum() if has_pnl else 0
        savg = spnl / sn
        print(f"    {shard:30s}  {sn:>5,} trades  WR={swr:5.1f}%  PnL=${spnl:>10,.2f}  avg=${savg:.2f}")

    # Output to file
    out_path = os.path.join('reports', 'findings', f'shard_summary_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"Shard Summary: {n:,} trades, WR={wr:.1f}%, PnL=${pnl:,.2f}, Avg=${avg:.2f}\n")
    print(f"\n  Summary saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Read IS/OOS signal log shards')
    parser.add_argument('--mode', default='is', choices=['is', 'oos'],
                        help='Which shards to read (default: is)')
    parser.add_argument('--depth', action='store_true',
                        help='Show per-depth breakdown')
    parser.add_argument('--exits', action='store_true',
                        help='Show exit reason breakdown')
    args = parser.parse_args()

    print(f"Shard Reader — {args.mode.upper()} mode")
    df = read_shards(args.mode)
    if df.empty:
        return

    print_summary(df, show_depth=args.depth, show_exits=args.exits)


if __name__ == '__main__':
    main()

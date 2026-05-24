"""Parity check -- live SIM vs forward pass (PRICE axis).

The forward pass / mock validated the engine assuming ZERO slippage
(fill = requested_price). This tool re-prices a live run's trade log at
requested_price to reconstruct the forward-pass-equivalent P&L, then diffs
it against the actual live P&L. The gap is pure execution slippage.

This covers PRICE parity only -- it assumes the same trade DECISIONS (it
re-prices the trades the live engine actually took). Decision parity (did
the live engine take the same trades the forward pass would) is a separate
check.

Input : reports/live/v2_trades_YYYY_MM_DD.csv
        columns: timestamp,type,tier,direction,requested_price,fill_price,
                 slippage,pnl,bars_held,exit_reason,is_chain,contracts,daily_pnl
Output: reports/live/parity_YYYY_MM_DD.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOLLAR_PER_POINT = 2.0          # MNQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='2026_05_20')
    ap.add_argument('--trades', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    trades_path = args.trades or f'reports/live/v2_trades_{args.date}.csv'
    out_path = args.out or f'reports/live/parity_{args.date}.txt'

    df = pd.read_csv(trades_path)

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    # ---- pair FILL_ENTRY / FILL_EXIT into round-trip trades ----------------
    trades = []
    open_e = None
    last_reason = ''
    for r in df.itertuples(index=False):
        if r.type == 'EXIT':
            last_reason = r.exit_reason if isinstance(r.exit_reason, str) else ''
        elif r.type == 'FILL_ENTRY':
            open_e = dict(ts=int(r.timestamp), dir=str(r.direction),
                          req=float(r.requested_price), fill=float(r.fill_price))
        elif r.type == 'FILL_EXIT':
            if open_e is None:
                continue
            sign = 1.0 if open_e['dir'] == 'long' else -1.0
            ereq, efill = float(r.requested_price), float(r.fill_price)
            live = sign * (efill - open_e['fill']) * DOLLAR_PER_POINT
            fp = sign * (ereq - open_e['req']) * DOLLAR_PER_POINT
            trades.append(dict(
                entry_ts=open_e['ts'], exit_ts=int(r.timestamp), dir=open_e['dir'],
                reason=last_reason,
                entry_req=open_e['req'], entry_fill=open_e['fill'],
                exit_req=ereq, exit_fill=efill,
                entry_slip_pt=open_e['fill'] - open_e['req'],
                exit_slip_pt=efill - ereq,
                live=live, fp=fp, gap=live - fp))
            open_e = None

    if not trades:
        out('No completed trades found in ' + trades_path)
        Path(out_path).write_text('\n'.join(lines), encoding='utf-8')
        return

    t = pd.DataFrame(trades)
    n = len(t)
    tot_live, tot_fp = t['live'].sum(), t['fp'].sum()
    gap = tot_live - tot_fp

    out('=' * 78)
    out(f'PRICE PARITY  --  live SIM vs forward pass (zero-slip)   {args.date}')
    out('=' * 78)
    out(f'Source: {trades_path}')
    out(f'Completed round-trip trades: {n}')
    out('')
    out(f'  Forward-pass P&L (fills @ requested_price) : ${tot_fp:+,.1f}')
    out(f'  Live P&L         (fills @ actual fill)     : ${tot_live:+,.1f}')
    out(f'  -------------------------------------------------------')
    out(f'  SLIPPAGE GAP (live - forward pass)         : ${gap:+,.1f}')
    out(f'  per trade: ${gap/n:+.1f}/trade   ({n} trades)')
    out('')
    if abs(tot_fp) > 1e-6:
        out(f'  The forward pass would have scored ${tot_fp:+,.0f}; live scored '
            f'${tot_live:+,.0f}.')
        out(f'  Slippage accounts for {gap:+,.0f} of the difference.')
    out('')

    # ---- slippage decomposition ------------------------------------------
    out('-' * 78)
    out('SLIPPAGE DECOMPOSITION')
    out('-' * 78)
    out(f'  entry slippage: mean {t["entry_slip_pt"].mean():+.2f} pt   '
        f'abs-mean {t["entry_slip_pt"].abs().mean():.2f} pt   '
        f'max abs {t["entry_slip_pt"].abs().max():.2f} pt')
    out(f'  exit  slippage: mean {t["exit_slip_pt"].mean():+.2f} pt   '
        f'abs-mean {t["exit_slip_pt"].abs().mean():.2f} pt   '
        f'max abs {t["exit_slip_pt"].abs().max():.2f} pt')
    out('')
    out('  gap by exit reason:')
    for reason, g in t.groupby('reason'):
        out(f'    {reason or "(none)":<18} n={len(g):>3}   '
            f'gap ${g["gap"].sum():+8.1f}   mean ${g["gap"].mean():+7.1f}/trade   '
            f'exit-slip abs-mean {g["exit_slip_pt"].abs().mean():.1f} pt')
    out('')

    # ---- worst trades -----------------------------------------------------
    out('-' * 78)
    out('LARGEST SLIPPAGE TRADES (by |gap|)')
    out('-' * 78)
    big = t.reindex(t['gap'].abs().sort_values(ascending=False).index).head(8)
    for r in big.itertuples(index=False):
        out(f'  {r.dir:<5} {r.reason:<16} '
            f'entry {r.entry_req:.2f}->{r.entry_fill:.2f} ({r.entry_slip_pt:+.2f}pt)  '
            f'exit {r.exit_req:.2f}->{r.exit_fill:.2f} ({r.exit_slip_pt:+.2f}pt)  '
            f'gap ${r.gap:+.0f}')
    out('')

    # ---- verdict ----------------------------------------------------------
    out('=' * 78)
    out('VERDICT (price parity)')
    out('=' * 78)
    n_big = int((t['gap'].abs() >= 20).sum())
    out(f'  {n_big}/{n} trades have a slippage gap >= $20.')
    out(f'  Total slippage drag: ${gap:+,.0f} over {n} trades (${gap/n:+.1f}/trade).')
    out('  NOTE: this is PRICE parity only -- it re-prices the trades the live')
    out('  engine actually took. Whether the engine took the SAME trades the')
    out('  forward pass would (decision parity) is a separate check.')

    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {out_path}')


if __name__ == '__main__':
    main()

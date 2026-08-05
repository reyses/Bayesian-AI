#!/usr/bin/env python3
"""CAUSAL EXIT-POLICY SWEEP (owner 2026-07-27): the leg peaks ~+43 pts near bar 7
then the R-trigger holds through it and gives it all back. Owner's rule: once
we're UP near the peak, START LOOKING FOR THE EXIT — arm a harvest once favorable
excursion reaches a target, rather than waiting for the ATR*4 reversal.

Re-runs the full-ATLAS strategy one-position-at-a-time (same sequencing as the
ledger: enter on dossier entry flag when flat) but swaps the EXIT:
  RIDE      : hold to R-trigger reversal / session close (never-bail baseline)
  TP{T}     : exit the first bar favorable excursion >= T pts (else ride)
  ARM{T}TR{r}: ride until fav >= T, then trail — exit on r-pt retrace from peak
              (else ride to R-trigger). Keeps never-bail on losers (no stop).
Causal, 1m close, per-day. Day-block bootstrap CI on $/day (CLAUDE.md discipline).
CPU-only. reports/exit_policy_sweep.md
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
VEC = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
ATLAS1M = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT_MD = os.path.join(PROJ, 'reports', 'exit_policy_sweep.md')

PT_USD, COMM = 2.0, 1.0
POLICIES = (['RIDE']
            + [f'TP{t}' for t in (20, 30, 40, 50)]
            + [f'ARM{t}TR{r}' for t in (30, 40) for r in (6, 10, 15)])


def exit_captured(fav, policy):
    """fav = dir-signed favorable excursion array over the causal hold window
    (fav[0]=0). Return (captured_pts, exit_bar)."""
    n = len(fav)
    if policy == 'RIDE':
        return fav[-1], n - 1
    if policy.startswith('TP'):
        T = float(policy[2:])
        for j in range(1, n):
            if fav[j] >= T:
                return fav[j], j
        return fav[-1], n - 1
    if policy.startswith('ARM'):
        T = float(policy[3:policy.index('TR')])
        r = float(policy[policy.index('TR') + 2:])
        armed = False; peak = 0.0
        for j in range(1, n):
            if fav[j] > peak:
                peak = fav[j]
            if not armed and fav[j] >= T:
                armed = True
            if armed and peak - fav[j] >= r:
                return fav[j], j
        return fav[-1], n - 1
    raise ValueError(policy)


def main():
    trades = {p: [] for p in POLICIES}       # (day, pnl_usd)
    for f in sorted(glob.glob(os.path.join(VEC, '*.parquet'))):
        day = os.path.basename(f)[:10]
        v = pd.read_parquet(f, columns=['bar_ts', 'entry', 'entry_dir', 'zz_confirm']).sort_values('bar_ts')
        p1 = os.path.join(ATLAS1M, f'{day}.parquet')
        if not os.path.exists(p1):
            continue
        m = pd.read_parquet(p1, columns=['timestamp', 'close'])
        cl = dict(zip(m['timestamp'].astype('int64'), m['close'].astype(float)))
        bts = v['bar_ts'].astype('int64').to_numpy()
        closes = np.array([cl.get(int(t), np.nan) for t in bts])
        ent = v['entry'].to_numpy(); edir = v['entry_dir'].to_numpy()
        zc = v['zz_confirm'].to_numpy()
        # each policy sequences independently (early exits let it re-enter sooner)
        for p in POLICIES:
            i = 0
            while i < len(bts):
                if np.isnan(closes[i]) or not (ent[i] == 1):
                    i += 1
                    continue
                d = int(edir[i]); e = closes[i]
                # causal hold ceiling = next R-trigger reversal or session end
                end = len(bts) - 1
                for j in range(i + 1, len(bts)):
                    if not np.isnan(closes[j]) and int(zc[j]) == -d:
                        end = j
                        break
                path = closes[i:end + 1]
                mask = ~np.isnan(path)
                path = path[mask]
                if len(path) < 2:
                    i += 1
                    continue
                fav = d * (path - path[0])
                cap, xbar = exit_captured(fav, p)
                trades[p].append((day, cap * PT_USD - COMM))
                # advance to the bar after the exit (map xbar back to bts index)
                valid_idx = np.where(mask)[0]
                i = i + int(valid_idx[xbar]) + 1
    rng = np.random.default_rng(42)

    def stats(rec):
        df = pd.DataFrame(rec, columns=['day', 'pnl'])
        net = df['pnl'].sum()
        bd = df.groupby('day')['pnl'].sum()
        boot = [rng.choice(bd.values, len(bd), replace=True).mean() for _ in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        w = df[df['pnl'] > 0]['pnl'].sum(); l = -df[df['pnl'] < 0]['pnl'].sum()
        return dict(n=len(df), net=net, perday=bd.mean(), lo=lo, hi=hi,
                    pf=w / l if l else float('nan'), sig=lo > 0)

    res = {p: stats(trades[p]) for p in POLICIES}
    order = sorted(POLICIES, key=lambda p: -res[p]['net'])
    lines = ['# Causal exit-policy sweep (full ATLAS, dossier entries)',
             f'{res["RIDE"]["n"]:,} RIDE trades, {pd.DataFrame(trades["RIDE"])[0].nunique()} days. '
             'Same entries; exit swapped. Never-bail on losers kept (no stop).',
             '',
             '| policy | trades | net $ | $/day | 95% CI | sig | PF |',
             '|---|---|---|---|---|---|---|']
    for p in order:
        r = res[p]
        lines.append(f"| {p} | {r['n']:,} | ${r['net']:,.0f} | ${r['perday']:.1f} "
                     f"| [${r['lo']:.1f}, ${r['hi']:.1f}] | "
                     f"{'**YES**' if r['sig'] else 'no'} | {r['pf']:.3f} |")
    lines += ['',
              'RIDE = current R-trigger never-bail baseline (~$16k / +$31/day, not '
              'sig). TP/ARM harvest the ~+43pt peak causally. A policy that beats '
              'RIDE with a CI excluding 0 is a real exit edge; if none do, the peak '
              'is not causally capturable (bar@MFE varies too much to arm on).']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

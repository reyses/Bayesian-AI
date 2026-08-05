#!/usr/bin/env python3
"""LEG-BODY ENTRY (owner 2026-07-28, TG): "just enter randomly but make sure you're
on the BODY of a leg." Direction is trivial once price commits; the cubic slope is
the leg-body detector (steep = in the body, flat = near a pivot). Test: enter in
sign(cubic_slope) when |slope|>=THR, ride until the slope FLIPS (leg ends). Random
timing, NO combiner. Controls: ANTIBODY (enter AGAINST the slope) and RANDOM dir —
if BODY >> RANDOM, being on the leg's SIDE is the edge; entry selection unneeded.
ATLAS 1m + cubic 7.5m slope. Friction 0.89pt/RT. reports/cubic_body_entry.md
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
import cubic_regression as cub                        # noqa: E402
A1 = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
A5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'edge_probe', 'reports', 'cubic_body_entry.md')
THR = [5, 10, 20, 40]                # |slope| pts/min to call "leg body"
FR = 0.89                            # friction pts/RT (~$1.78 MNQ)
LAST = 150
rng = np.random.default_rng(42)


def run(days, thr, mode):
    """one-position: enter when flat & |slope|>=thr (dir by mode), exit when slope
    flips (dir*slope<=0). Returns list of (day, pnl_pts)."""
    tr = []
    for day, cl, slp in days:
        n = len(cl); i = 0
        while i < n:
            if np.isnan(cl[i]) or np.isnan(slp[i]) or abs(slp[i]) < thr:
                i += 1; continue
            if mode == 'body':
                d = int(np.sign(slp[i]))
            elif mode == 'anti':
                d = -int(np.sign(slp[i]))
            else:
                d = int(rng.choice([-1, 1]))
            e = cl[i]; j = i + 1
            while j < n and not (not np.isnan(slp[j]) and d * slp[j] <= 0):
                j += 1
            j = min(j, n - 1)
            if not np.isnan(cl[j]):
                tr.append((day, d * (cl[j] - e) - FR))
            i = j + 1
    return tr


def stats(tr):
    df = pd.DataFrame(tr, columns=['day', 'pnl'])
    bd = df.groupby('day')['pnl'].sum()
    boot = [rng.choice(bd.values, len(bd), True).mean() for _ in range(3000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    w = df[df.pnl > 0].pnl.sum(); l = -df[df.pnl < 0].pnl.sum()
    return (len(df), df.pnl.mean(), bd.mean(), lo, hi, (w / l - 1) if l else float('nan'))


def main():
    files = sorted(glob.glob(os.path.join(A1, '*.parquet')))[-LAST:]
    days = []
    for f in files:
        day = os.path.basename(f)[:10]; p5 = os.path.join(A5, f'{day}.parquet')
        if not os.path.exists(p5):
            continue
        m = pd.read_parquet(f, columns=['timestamp', 'close'])
        d5 = pd.read_parquet(p5, columns=['timestamp', 'close']).sort_values('timestamp')
        t5 = d5['timestamp'].astype('int64').to_numpy()
        _, cslp, _ = cub.rolling(d5['close'].astype(float).to_numpy(), 90, 5)
        ts = m['timestamp'].astype('int64').to_numpy(); cl = m['close'].astype(float).to_numpy()
        idx = np.searchsorted(t5, ts, side='right') - 1
        slp = np.where(idx >= 0, cslp[np.clip(idx, 0, len(cslp) - 1)], np.nan)
        days.append((day, cl, slp))
    lines = ['# Leg-body entry — cubic-slope direction vs random (ATLAS 1m)', '',
             f'{len(days)} days. Enter sign(slope) when |slope|>=THR, ride till slope flips. '
             f'Friction {FR}pt/RT.', '',
             '| THR (pt/min) | mode | N | mean pt/trade | $/day (×$2) | 95% CI | PF-WR |',
             '|---|---|---|---|---|---|---|']
    for thr in THR:
        for mode in ['body', 'anti', 'rand']:
            n, mpt, pd_, lo, hi, pfwr = stats(run(days, thr, mode))
            lines.append(f'| {thr} | {mode} | {n:,} | {mpt:+.2f} | {pd_*2:+.1f} '
                         f'| [{lo*2:+.1f}, {hi*2:+.1f}] | {pfwr:+.3f} |')
    lines += ['',
              'Read: if BODY mean pt/trade > 0 and >> ANTI/RAND, being on the leg\'s SIDE '
              '(cubic-slope direction) is a real edge with random timing = "just be in a leg" '
              'holds, and the cubic is the leg-body+direction detector. If BODY ≈ RAND ≈ 0 '
              'after friction, leg-body entry has no edge (the slope is not causally '
              'predictive of the continuation).']
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

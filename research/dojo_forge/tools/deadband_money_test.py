#!/usr/bin/env python
"""DEAD-BAND TURN SIGNAL -- THE MONEY TEST (owner 2026-07-31: "run a sweep for
the money test").

Strategy under test, deliberately minimal so the SIGNAL is what's priced:
  flip-on-turn, always in market. At each CONFIRMED turn of the dead-banded
  cubic line, flip to the line's new direction at that bar's close. P&L per
  segment = close-to-close move in the held direction minus one round-trip
  friction. No stops, no targets, no management -- those live in other layers;
  this prices the turn signal itself.

Causality: cubic endpoint is causal; dead-band is causal; a turn is only KNOWN
when the line retraces `confirm` pts off its running extreme -- fills happen at
the CONFIRMATION bar's close, never at the turn's extremum.

Sweep: D (dead-band pts) x confirm (hysteresis pts). D=0 = raw cubic baseline
(expected friction pump: ~491 turns/day). Metrics per project rules: $/trade
with 4000-resample bootstrap CI + significance, PF-based Trade WR, $/day, N.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_ground_truth import D5, FRICTION_PT, PT_USD
import cubic_regression as _cub

BOOT = 4000     # bootstrap resamples (project standard)
SEED = 11


def deadband(line, D):
    if D <= 0: return line
    out = np.copy(line)
    for i in range(1, len(line)):
        diff = line[i] - out[i - 1]
        out[i] = out[i - 1] + np.sign(diff) * max(0.0, abs(diff) - D)
    return out


def turn_bars(line, confirm):
    """Causal confirmed turns: bar index where the line has retraced `confirm`
    pts from its running extreme. Returns (bars, new_direction)."""
    idx, dirs = [], []
    d, ext = 0, line[0]
    for i in range(1, len(line)):
        v = line[i]
        if not np.isfinite(v): continue
        if d >= 0:
            if v > ext: ext = v
            if ext - v >= confirm:
                if d != 0: idx.append(i); dirs.append(-1)
                d, ext = -1, v
                continue
        if d <= 0:
            if v < ext: ext = v
            if v - ext >= confirm:
                if d != 0: idx.append(i); dirs.append(1)
                d, ext = 1, v
    return idx, dirs


def boot_ci(x, n=BOOT, seed=SEED):
    a = np.asarray(x, float)
    if len(a) < 2: return (float('nan'),) * 2
    rng = np.random.default_rng(seed)
    s = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def pf_wr(pts):
    a = np.asarray(pts, float)
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    return (w / l - 1.0) if l > 0 else float('inf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--Ds', default='0,5,10,15,20,25')
    ap.add_argument('--confirms', default='2,5')
    a = ap.parse_args()
    Ds = [float(x) for x in a.Ds.split(',')]
    Cs = [float(x) for x in a.confirms.split(',')]
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet'))[-a.days:]

    cub_cache = {}
    for day in tqdm(days, desc='cubic'):
        df = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))
        if len(df) < 2000: continue
        cl = df['close'].to_numpy()
        cub, _, _ = _cub.rolling(cl, 90, 5)
        cub_cache[day] = (np.where(np.isfinite(cub), cub, cl), cl)

    print(f"\nFLIP-ON-TURN MONEY TEST  ({len(cub_cache)} days, fill = confirmation bar close,"
          f" friction {FRICTION_PT}pt/round-trip)\n")
    h = (f"{'config':>16} {'trades/d':>9} {'$/trade':>9} {'95% CI':>18} {'sig':>4} "
         f"{'tradeWR':>8} {'$/day':>9} {'$/day CI':>20}")
    print(h); print('-' * len(h))
    results = []
    for D in Ds:
        for C in Cs:
            pts, day_pnl = [], []
            for day, (cub, cl) in cub_cache.items():
                line = deadband(cub, D)
                bars, dirs = turn_bars(line, C)
                dp = 0.0
                for (b0, d0), b1 in zip(zip(bars, dirs), bars[1:] + [len(cl) - 1]):
                    p = (cl[b1] - cl[b0]) * d0 - FRICTION_PT
                    pts.append(p); dp += p
                day_pnl.append(dp)
            if not pts: continue
            lo, hi = boot_ci(pts)
            dlo, dhi = boot_ci(day_pnl)
            sig = 'YES' if (lo > 0 or hi < 0) else 'no'
            results.append((np.mean(pts), D, C))
            print(f"D={D:>4g} c={C:>3g} {len(pts)/len(cub_cache):>9.1f} "
                  f"{np.mean(pts)*PT_USD:>9.2f} [{lo*PT_USD:>+7.2f},{hi*PT_USD:>+7.2f}] {sig:>4} "
                  f"{pf_wr(pts):>8.3f} {np.mean(day_pnl)*PT_USD:>9.0f} "
                  f"[{dlo*PT_USD:>+8.0f},{dhi*PT_USD:>+8.0f}]")
    best = max(results)
    print(f"\n  best $/trade: D={best[1]:g} confirm={best[2]:g}")


if __name__ == '__main__':
    main()

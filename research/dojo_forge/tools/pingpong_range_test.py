#!/usr/bin/env python
"""Ping-pong between FIXED churn extremes — the owner's actual design.

WHY (owner, 2026-08-01): "the watcher needs to wait until it reaches an extreme
from the churn perspective, and enter betting on the return to the center of the
mean, and overshoot to the other extreme; for it to work we need to set levels
at both extremes and then ping pong on the relative levels."

THIS IS NOT THE TEST ALREADY RUN. `oscillation_harvest_test.py` faded a ±1.5σ
cubic band — and the bands MOVE, because the cubic endpoint is a ~zero-lag
tracker. That is why its "traverse completes 78.5%" overstated real reversion: a
traverse can complete because the MEAN came to the price. The owner's design
uses FIXED PRICE levels taken from the observed churn extremes, which cannot
chase price. He arrived at the correction for that flaw independently.

DESIGN AS SPECIFIED:
  1. observe a churn window and take its extremes as HARD levels (H, L)
  2. wait for price to reach one of them
  3. enter fading, target the OPPOSITE level (the overshoot, not the middle)
  4. stop outside the range
  5. repeat — ping-pong

The range must be built from bars STRICTLY BEFORE the touch, or the level is
fitted to the move it is supposed to predict.

Compared head-to-head against the moving-band version on identical entries so
the only difference is fixed-vs-moving targets.

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/pingpong_range_test.py --exclude 2024_09_16
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
import cubic_regression as _cub                                    # noqa: E402

D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'pingpong_range.md')

CUBIC_5S_WINDOW = 90
SIGMA_MIN = 20
BAND = 1.5
RANGE_WIN_S = 1800      # churn window the fixed levels are taken from (30min)
TOUCH_TOL = 1.0         # pt; "reached the level"
STOP_BEYOND = 8.0       # stop this far OUTSIDE the range — a range trade dies
                        # when the range dies, so the stop belongs at the level
MAX_HOLD_S = 3600
MIN_RANGE, MAX_RANGE = 8.0, 60.0    # ignore degenerate and non-range regimes
RTH_FROM, RTH_TO = 570, 960
FRICTION_PT = 0.89
PT_USD = 2.0
BOOT = 4000
SEED = 11


def ci(x):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    if len(x) < 5:
        return float('nan'), float('nan')
    return tuple(np.percentile(
        [rng.choice(x, len(x), replace=True).mean() for _ in range(BOOT)],
        [2.5, 97.5]))


def scan_day(day):
    d = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))[
        ['timestamp', 'high', 'low', 'close']]
    if len(d) < 2000:
        return []
    ts = d['timestamp'].to_numpy(); c = d['close'].to_numpy()
    hi = d['high'].to_numpy(); lo = d['low'].to_numpy()
    cub, _, _ = _cub.rolling(c, CUBIC_5S_WINDOW, 5)
    res = c - cub
    sig = pd.Series(res).rolling(SIGMA_MIN * 12, min_periods=5 * 12).std().to_numpy()
    z = np.where(sig > 0, res / sig, np.nan)
    e = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    m = (e.hour * 60 + e.minute).to_numpy()
    side = np.where(z >= BAND, 1, np.where(z <= -BAND, -1, 0))
    ff = pd.Series(np.where(side == 0, np.nan, side)).ffill().to_numpy()
    flip = np.flatnonzero((~np.isnan(ff[1:])) & (~np.isnan(ff[:-1]))
                          & (ff[1:] != ff[:-1])) + 1
    flip = flip[(m[flip] >= RTH_FROM) & (m[flip] < RTH_TO)]

    rows = []
    w_bars = RANGE_WIN_S // 5
    for i in flip:
        if i < w_bars:
            continue
        K = int(((ts[flip] >= ts[i] - RANGE_WIN_S) & (ts[flip] < ts[i])).sum())
        if K < 5:
            continue                       # churn regime only, as specified
        # levels from bars STRICTLY BEFORE the touch
        pw = slice(i - w_bars, i)
        H, L = float(hi[pw].max()), float(lo[pw].min())
        width = H - L
        if not (MIN_RANGE <= width <= MAX_RANGE):
            continue
        p0 = float(c[i])
        near_lo = abs(p0 - L) <= max(TOUCH_TOL, 0.15 * width)
        near_hi = abs(p0 - H) <= max(TOUCH_TOL, 0.15 * width)
        if near_lo == near_hi:
            continue                       # must be AT one extreme, not both/neither
        sgn = 1 if near_lo else -1
        tgt = H if sgn > 0 else L                       # the OPPOSITE extreme
        stop = (L - STOP_BEYOND) if sgn > 0 else (H + STOP_BEYOND)

        w = (ts > ts[i]) & (ts <= ts[i] + MAX_HOLD_S)
        if w.sum() < 12:
            continue
        hh, ll, cc, tt, zz = hi[w], lo[w], c[w], ts[w], z[w]
        jt = np.flatnonzero(hh >= tgt) if sgn > 0 else np.flatnonzero(ll <= tgt)
        js = np.flatnonzero(ll <= stop) if sgn > 0 else np.flatnonzero(hh >= stop)
        jt = jt[0] if len(jt) else None
        js = js[0] if len(js) else None
        if jt is not None and (js is None or jt < js):
            out, pts = 'target', float((tgt - p0) * sgn)
        elif js is not None:
            out, pts = 'stopped', float((stop - p0) * sgn)
        else:
            out, pts = 'timeout', float((cc[-1] - p0) * sgn)
        # head-to-head: the MOVING-band exit on the SAME entry
        bt = np.flatnonzero(zz >= BAND) if sgn > 0 else np.flatnonzero(zz <= -BAND)
        bs = np.flatnonzero(ll <= p0 - 20) if sgn > 0 else np.flatnonzero(hh >= p0 + 20)
        bt = bt[0] if len(bt) else None
        bs = bs[0] if len(bs) else None
        if bt is not None and (bs is None or bt < bs):
            band_pts = float((cc[bt] - p0) * sgn)
        elif bs is not None:
            band_pts = -20.0
        else:
            band_pts = float((cc[-1] - p0) * sgn)
        rows.append(dict(day=day, width=width, outcome=out, pts=pts,
                         net=pts - FRICTION_PT,
                         band_net=band_pts - FRICTION_PT))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', nargs='*', default=['2024_09_16'])
    ap.add_argument('--days', type=int, default=200)
    a = ap.parse_args()
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    if len(days) > a.days:
        days = sorted(rng.choice(days, a.days, replace=False).tolist())
    rows = []
    for d in tqdm(days, desc='pingpong'):
        try:
            rows += scan_day(d)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        print('no rows'); return

    net = df['net'].to_numpy(); bnet = df['band_net'].to_numpy()
    lo, hi = ci(net); blo, bhi = ci(bnet)
    # PAIRED — same entries, only the exit differs
    d_ = net - bnet
    dlo, dhi = ci(d_)

    L = ['# Ping-pong between FIXED churn extremes (owner design)', '',
         'Levels are taken from the observed churn window and do NOT move. This '
         'is the correction to the moving-σ-band version, whose "traverse '
         'completes" rate was inflated by the cubic chasing price.', '',
         f'Range from the prior {RANGE_WIN_S // 60}min (bars strictly BEFORE the '
         f'touch), width {MIN_RANGE:g}–{MAX_RANGE:g}pt, K≥5 churn regime, '
         f'entry at one extreme, target the OPPOSITE extreme, stop '
         f'{STOP_BEYOND:g}pt outside the range, max hold {MAX_HOLD_S // 60}min, '
         f'friction {FRICTION_PT}pt.',
         f'Excluded: {", ".join(a.exclude)}.', '',
         f'**N = {len(df)} trades across {df["day"].nunique()} sessions.** '
         f'Median range width {df["width"].median():.1f}pt.', '',
         '## Outcome', '',
         f'- target hit **{(df["outcome"] == "target").mean():.1%}** · '
         f'stopped {(df["outcome"] == "stopped").mean():.1%} · '
         f'timeout {(df["outcome"] == "timeout").mean():.1%}',
         f'- mean net **{net.mean():+.2f}pt** (${net.mean() * PT_USD:+.2f}) '
         f'95% CI `[{lo:+.2f}, {hi:+.2f}]` → '
         f'{"NOT significant" if lo <= 0 <= hi else ("SIGNIFICANTLY POSITIVE" if lo > 0 else "significantly negative")}',
         f'- median net {np.median(net):+.2f}pt', '',
         '## Head-to-head vs the moving band, same entries (PAIRED)', '',
         f'- fixed-extreme exit: **{net.mean():+.2f}pt** `[{lo:+.2f}, {hi:+.2f}]`',
         f'- moving-band exit:   **{bnet.mean():+.2f}pt** `[{blo:+.2f}, {bhi:+.2f}]`',
         f'- paired Δ (fixed − band): **{d_.mean():+.2f}pt** '
         f'95% CI `[{dlo:+.2f}, {dhi:+.2f}]` → '
         f'{"NOT significant" if dlo <= 0 <= dhi else ("fixed levels WIN" if dlo > 0 else "fixed levels LOSE")}',
         '', 'The paired comparison is the point: identical entries, identical '
             'regime filter, only the exit definition differs.', '']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
